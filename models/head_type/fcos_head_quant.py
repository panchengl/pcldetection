import torch.nn as nn
import torch
import math
from models.quant_type.quant_dorefa import Conv2d_dorefa

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)



class class_centerness_regression_head(nn.Module):
    def __init__(self, fpn_output_channels=256, class_num=80,  output_channels=256,  grounp_norm=True, prior=0.01):
        super(class_centerness_regression_head, self).__init__()
        self.cnt_on_reg = True
        self.fpn_output_channels = fpn_output_channels
        self.class_num = class_num
        self.prior = prior
        self.output_channels = 256
        self.grounp_norm = grounp_norm
        self.regression_channels = output_channels
        self.class_channels = output_channels
        self.center_ness_channels = output_channels

        self.regression_logits = self._make_loc_head(head_output_channels=self.regression_channels, grounp_norm=self.grounp_norm)
        self.classification_logits = self._make_cls_head( head_output_channels=self.class_channels, grounp_norm=self.grounp_norm)
        self.regression_head = Conv2d_dorefa(output_channels, 4, kernel_size=3, stride=1, padding=1)
        self.center_ness_head = Conv2d_dorefa(output_channels, 1, kernel_size=3, stride=1, padding=1)
        self.classification_head = Conv2d_dorefa(output_channels, class_num, kernel_size=3, stride=1, padding=1)

        self.apply(self.init_conv_RandomNormal)
        nn.init.constant_(self.classification_head.bias, -math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])


    def _make_loc_head(self, head_output_channels, grounp_norm=True):
        layers = []
        for _ in range(4):
            layers.append(Conv2d_dorefa(256, head_output_channels, kernel_size=3, stride=1, padding=1))
            if grounp_norm:
                layers.append(nn.GroupNorm(32, head_output_channels))
            layers.append(nn.ReLU(True))
            layers.append(Conv2d_dorefa(head_output_channels, head_output_channels, kernel_size=3, stride=1, padding=1))
            if grounp_norm:
                layers.append(nn.GroupNorm(32, head_output_channels))
            layers.append(nn.ReLU(True))
        # in this head, in order to create centerness head, so, last conv don`t write in this function
        # layers.append(Conv2d_dorefa(head_output_channels, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def _make_cls_head(self, head_output_channels, grounp_norm=True):
        layers = []
        for _ in range(4):
            layers.append(Conv2d_dorefa(256, head_output_channels, kernel_size=3, stride=1, padding=1))
            if grounp_norm:
                layers.append(nn.GroupNorm(32, head_output_channels))
            layers.append(nn.ReLU(True))
            layers.append(Conv2d_dorefa(head_output_channels, head_output_channels, kernel_size=3, stride=1, padding=1))
            if grounp_norm:
                layers.append(nn.GroupNorm(32, head_output_channels))
            layers.append(nn.ReLU(True))
        # in this head, in order to create centerness head, so, last conv don`t write in this function
        # layers.append(Conv2d_dorefa(head_output_channels, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, Conv2d_dorefa):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        '''inputs:[P3~P7]'''
        class_preds = []
        centerness_preds = []
        regession_preds = []
        for index, p_feature in enumerate(inputs):
            class_logits = self.classification_logits(p_feature)
            reg_logits = self.regression_logits(p_feature)

            # class_preds.append(self.classification_head(class_logits).permute(0, 2, 3, 1).contiguous())
            # regession_preds.append(self.scale_exp[index](self.regression_head(reg_logits).permute(0, 2, 3, 1).contiguous()))
            # if not self.cnt_on_reg:
            #     centerness_preds.append(self.center_ness_head(class_logits).permute(0, 2, 3, 1).contiguous())
            # else:
            #     centerness_preds.append(self.center_ness_head(reg_logits).permute(0, 2, 3, 1).contiguous())

            class_preds.append(self.classification_head(class_logits))
            regession_preds.append(self.scale_exp[index](self.regression_head(reg_logits)))
            if not self.cnt_on_reg:
                centerness_preds.append(self.center_ness_head(class_logits))
            else:
                centerness_preds.append(self.center_ness_head(reg_logits))
        return class_preds, centerness_preds, regession_preds
