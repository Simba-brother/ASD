import random
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
from torch.utils.data import DataLoader

from .backdoor import BadNets,BadNets_2
from .cifar import CIFAR10
from .prefetch import PrefetchLoader


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.
    
    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


def query_transform(name, kwargs):
    if name == "random_crop":
        return transforms.RandomCrop(**kwargs)
    elif name == "random_resize_crop":
        return transforms.RandomResizedCrop(**kwargs)
    elif name == "resize":
        return transforms.Resize(**kwargs)
    elif name == "center_crop":
        return transforms.CenterCrop(**kwargs)
    elif name == "random_horizontal_flip":
        return transforms.RandomHorizontalFlip(**kwargs)
    elif name == "random_color_jitter":
        # In-place!
        p = kwargs.pop("p")
        return transforms.RandomApply([transforms.ColorJitter(**kwargs)], p=p)
    elif name == "random_grayscale":
        return transforms.RandomGrayscale(**kwargs)
    elif name == "gaussian_blur":
        # In-place!
        p = kwargs.pop("p")
        return transforms.RandomApply([GaussianBlur(**kwargs)], p=p)
    elif name == "to_tensor":
        if kwargs:
            return transforms.ToTensor()
    elif name == "normalize":
        return transforms.Normalize(**kwargs)
    else:
        raise ValueError("Transformation {} is not supported!".format(name))


def get_transform(transform_config):
    transform = []
    if transform_config is not None:
        for (k, v) in transform_config.items():
            if v is not None:
                transform.append(query_transform(k, v))
    transform = transforms.Compose(transform)

    return transform


def get_dataset(dataset_dir, transform, train=True, prefetch=False):
    if "cifar" in dataset_dir:
        dataset = CIFAR10(  # 自己写的class
            dataset_dir, transform=transform, train=train, prefetch=prefetch
        )
    else:
        raise ValueError("Dataset in {} is not supported.".format(dataset_dir))

    return dataset


def get_loader(dataset, loader_config=None, **kwargs):
    if loader_config is None:
        loader = DataLoader(dataset, **kwargs)
    else:
        loader = DataLoader(dataset, **loader_config, **kwargs)
    if dataset.prefetch:
        loader = PrefetchLoader(loader, dataset.mean, dataset.std)

    return loader


def gen_poison_idx(dataset, target_label, poison_ratio=None):
    poison_idx = np.zeros(len(dataset))
    train = dataset.train
    for (i, t) in enumerate(dataset.targets):
        if train and poison_ratio is not None:
            # 只对训练集进行poison_ratio比例污染
            if random.random() < poison_ratio: # and t != target_label: 是否污染目标类别的样本，原始仓库是不污染目标类别的样本的。
                poison_idx[i] = 1
        else:
            if t != target_label:
                poison_idx[i] = 1
    return poison_idx


def get_bd_transform(bd_config):
    if "badnets" in bd_config:
        bd_transform = BadNets(bd_config["badnets"]["trigger_path"]) # ./data/trigger/cifar_2.png

        '''
        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[4:6, 4:6] = 255 # pattern[-3:, -3:] = 255 (右下角)，pattern[4:6, 4:6] = 255 (ASD trigger图像白点像素位置)
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[4:6, 4:6] = 1.0 # weight[-3:, -3:] = 255 (右下角)，weight[4:6, 4:6] = 255 (ASD trigger图像白点像素位置)
        bd_transform = BadNets_2(pattern, weight)
        '''
        
    else:
        raise ValueError("Backdoor {} is not supported.".format(bd_config))

    return bd_transform # 本质是一个BadNets类（data/backdoor.py/BadNets）的实例
