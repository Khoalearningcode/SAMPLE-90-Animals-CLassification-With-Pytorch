from torch import nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

def efficientnet_b0(len_classes=90, requires_grad=False):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for params in model.features.parameters():
        params.requires_grad = requires_grad
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                     nn.Linear(in_features=1280, out_features=len_classes))
    return model




