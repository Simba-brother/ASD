import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.backbone = backbone # 骨架
        self.linear = nn.Linear(feature_dim, num_classes) # 输出

    def forward(self, x):
        # 重写forward
        feature = self.backbone(x)
        out = self.linear(feature)
        return out

    def update_encoder(self, backbone):
        # 修改骨架
        self.backbone = backbone
