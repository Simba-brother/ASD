import numpy as np
from PIL import Image
import torch


class BadNets(object):
    """The BadNets [paper]_ backdoor transformation. Inject a trigger into an image (ndarray with
    shape H*W*C) to get a poisoned image (ndarray with shape H*W*C).

    Args:
        trigger_path (str): The path of trigger image whose background is in black.

    .. rubric:: Reference

    .. [paper] "Badnets: Evaluating backdooring attacks on deep neural networks."
     Tianyu Gu, et al. IEEE Access 2019.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB") # PIL.Image
        self.trigger_ptn = np.array(trigger_ptn) # PIL.Image to ndarray, shape:HWC
        self.trigger_loc = np.nonzero(self.trigger_ptn) # ndarray 中非0,Return the indices of the elements that are non-zero.
    
    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        '''
        添加trigger前图像必须是ndarray且shape必须==3(HWC)
        通俗来讲，给图片贴白，返回一个ndarray
        '''
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3: # H*W*C
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        img[self.trigger_loc] = 0 # 把图像中trigger非0值的对应位置像素设置为0
        poison_img = img + self.trigger_ptn # 把图像和trigger相加得到木马样本

        return poison_img # ndarray, HWC



class BadNets_2(object):
    """The BadNets [paper]_ backdoor transformation. Inject a trigger into an image (ndarray with
    shape H*W*C) to get a poisoned image (ndarray with shape H*W*C).

    Args:
        trigger_path (str): The path of trigger image whose background is in black.

    .. rubric:: Reference

    .. [paper] "Badnets: Evaluating backdooring attacks on deep neural networks."
     Tianyu Gu, et al. IEEE Access 2019.
    """

    def __init__(self,pattern, weight):
        self.pattern = pattern
        self.weight = weight

        if self.pattern.dim() == 2:
            self.pattern = self.pattern.unsqueeze(0)
        if self.weight.dim() == 2:
            self.weight = self.weight.unsqueeze(0)

        self.mask = self.weight * self.pattern
        self.weight = 1.0 - self.weight  # self.weight

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        '''
        img(ndarray, HWC):benign
        '''
        if not isinstance(img, np.ndarray):
            raise TypeError("Img should be np.ndarray. Got {}".format(type(img)))
        if len(img.shape) != 3: # H*W*C
            raise ValueError("The shape of img should be HWC. Got {}".format(img.shape))
        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1) # tensor
        poisoned_tensor = (self.weight * img + self.mask).type(torch.uint8)
        # CHW -> HWC, tensor -> ndarray
        poisoned_tensor_HWC = poisoned_tensor.permute(1,2,0)
        poisoned_array_HWC = poisoned_tensor_HWC.cpu().numpy()
        return poisoned_array_HWC