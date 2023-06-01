import torch.nn as nn
import torch.nn.functional as F
import torch
from roi import TorchROIPooling
from baseline_model import BaselineModel

device = 'cuda'

class BaselineROI(torch.nn.Module):
    def __init__(self):
        super(BaselineROI, self).__init__()

        # Convolution model
        self.conv_model = BaselineModel()
        
        # ROI model
        self.roi_model = TorchROIPooling(output_size=2).to(device)

        # fully connected fusion layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
        )

    def forward(self, x, proposals):
        conv_x = self.conv_model(x, proposals)
        roi_x = self.roi_model(x, proposals)

        # print(conv_x.shape, roi_x.shape)

        assert(conv_x.shape[1] == roi_x.shape[1])

        x = torch.hstack((conv_x, roi_x))
        x = self.fc_layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    










    # add label to global and local feature
    # def forward(self, x, proposals):
    #     b, _, _, _, _ = x.shape
    #     conv_x = self.conv_model(x, proposals)
    #     roi_x = self.roi_model(x, proposals)

    #     (b, dim1, 1) -> (b, dim1, 2)
    #     (b, dim2, 1) -> (b, dim2, 2)
    #     conv_x = conv_x.view((-1, conv_x.shape[1], 1))
    #     flg_global = torch.ones(conv_x.shape).to(device)

    #     roi_x = roi_x.view((-1,roi_x.shape[1], 1))
    #     flg_local = torch.zeros(roi_x.shape).to(device)

    #     conv_x = torch.concat([conv_x, flg_global], dim=-1)
    #     roi_x = torch.concat([roi_x, flg_local], dim=-1)

    #     conv_x = conv_x.reshape(b, -1)
    #     roi_x = roi_x.reshape(b, -1)

    #     x = torch.hstack((conv_x, roi_x))
    #     x = self.fc_layers(x)
    #     x = F.normalize(x, p=2, dim=1)
    #     return x
