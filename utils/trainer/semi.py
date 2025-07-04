"""Modified from https://github.com/YU1ut/MixMatch-pytorch.
"""

import time

import numpy as np
import torch

from .log import AverageMeter, Record, tabulate_step_meter, tabulate_epoch_meter


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch

    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    return [torch.cat(v, dim=0) for v in xy]


def mixmatch_train(
    model, xloader, uloader, criterion, optimizer, epoch, logger, **kwargs,
):
    loss_meter = AverageMeter("loss")
    xloss_meter = AverageMeter("xloss")
    uloss_meter = AverageMeter("uloss")
    lambda_u_meter = AverageMeter("lambda_u")
    meter_list = [loss_meter, xloss_meter, uloss_meter, lambda_u_meter]

    xiter = iter(xloader)
    uiter = iter(uloader)

    model.train()
    gpu = next(model.parameters()).device
    start = time.time()
    for batch_idx in range(kwargs["train_iteration"]): # 1024 个 batch
        try:
            xbatch = next(xiter)
            xinput, xtarget = xbatch["img"], xbatch["target"]
        except:
            xiter = iter(xloader)
            xbatch = next(xiter)
            xinput, xtarget = xbatch["img"], xbatch["target"]

        try:
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]
        except:
            uiter = iter(uloader)
            ubatch = next(uiter)
            uinput1, uinput2 = ubatch["img1"], ubatch["img2"]

        batch_size = xinput.size(0)
        # xtarget one-hot
        xtarget = torch.zeros(batch_size, kwargs["num_classes"]).scatter_(
            1, xtarget.view(-1, 1).long(), 1
        )
        xinput = xinput.cuda(gpu, non_blocking=True)
        xtarget = xtarget.cuda(gpu, non_blocking=True)
        uinput1 = uinput1.cuda(gpu, non_blocking=True)
        uinput2 = uinput2.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            uoutput1 = model(uinput1)
            uoutput2 = model(uinput2)
            p = (torch.softmax(uoutput1, dim=1) + torch.softmax(uoutput2, dim=1)) / 2
            pt = p ** (1 / kwargs["temperature"])
            utarget = pt / pt.sum(dim=1, keepdim=True)
            utarget = utarget.detach()


        all_input = torch.cat([xinput, uinput1, uinput2], dim=0)
        all_target = torch.cat([xtarget, utarget, utarget], dim=0)
        l = np.random.beta(kwargs["alpha"], kwargs["alpha"])
        l = max(l, 1 - l)
        idx = torch.randperm(all_input.size(0))
        input_a, input_b = all_input, all_input[idx]
        target_a, target_b = all_target, all_target[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logit = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logit.append(model(input))

        # put interleaved samples back
        logit = interleave(logit, batch_size)
        xlogit = logit[0]
        ulogit = torch.cat(logit[1:], dim=0)

        Lx, Lu, lambda_u = criterion(
            xlogit,
            mixed_target[:batch_size],
            ulogit,
            mixed_target[batch_size:],
            epoch + batch_idx / kwargs["train_iteration"],
        )
        loss = Lx + lambda_u * Lu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()

        loss_meter.update(loss.item())
        xloss_meter.update(Lx.item())
        uloss_meter.update(Lu.item())
        lambda_u_meter.update(lambda_u)
        tabulate_step_meter(batch_idx, kwargs["train_iteration"], 3, meter_list, logger)

    logger.info("MixMatch training summary:")
    tabulate_epoch_meter(time.time() - start, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result


def linear_test(model, loader, criterion, logger):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.eval()
    gpu = next(model.parameters()).device
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            output = model(data)
        criterion.reduction = "mean"
        loss = criterion(output, target)

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 2, meter_list, logger)

    logger.info("Linear test summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list, logger)
    result = {m.name: m.total_avg for m in meter_list}

    return result


def poison_linear_record(model, loader, criterion):
    num_data = len(loader.dataset)
    target_record = Record("target", num_data)
    poison_record = Record("poison", num_data)
    origin_record = Record("origin", num_data)
    loss_record = Record("loss", num_data)
    record_list = [
        target_record,
        poison_record,
        origin_record,
        loss_record,
    ]

    model.eval()
    # 判断模型是在CPU还是GPU上
    gpu = next(model.parameters()).device
    for _, batch in enumerate(loader):
        data = batch["img"].cuda(gpu, non_blocking=True)
        target = batch["target"].cuda(gpu, non_blocking=True)
        with torch.no_grad():
            '''
            feature = model.backbone(data)
            output = model.linear(feature)
            '''
            output = model(data)
        criterion.reduction = "none"
        raw_loss = criterion(output, target)

        target_record.update(batch["target"])
        poison_record.update(batch["poison"])
        origin_record.update(batch["origin"])
        loss_record.update(raw_loss.cpu())
    return record_list

