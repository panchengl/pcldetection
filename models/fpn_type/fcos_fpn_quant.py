import torch.nn as nn
from models.quant_type.quant_dorefa import Conv2d_dorefa
class FPN(nn.Module):
    def __init__(self,  c_input_channels, fpn_output_channels=256, use_p5 = True):
        super(FPN, self).__init__()
        c3_input_channels, c4_input_channels, c5_input_channels = c_input_channels
        self.use_p5 = use_p5
        self.p5_input = Conv2d_dorefa(c5_input_channels, fpn_output_channels, kernel_size=1, stride=1, padding=0)
        self.p5_unsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_output = Conv2d_dorefa(fpn_output_channels, fpn_output_channels, kernel_size=3, stride=1, padding=1)

        self.p4_input = Conv2d_dorefa(c4_input_channels, fpn_output_channels, kernel_size=1, stride=1,  padding=0)
        self.p4_unsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_output = Conv2d_dorefa(fpn_output_channels, fpn_output_channels, kernel_size=3, stride=1, padding=1)

        self.p3_input = Conv2d_dorefa(c3_input_channels, fpn_output_channels, kernel_size=1, stride=1, padding=0)
        self.p3_unsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_output = Conv2d_dorefa(fpn_output_channels, fpn_output_channels, kernel_size=3, stride=1, padding=1)
        if use_p5:
            self.p6_output = Conv2d_dorefa(fpn_output_channels, fpn_output_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.p6_output = Conv2d_dorefa(c5_input_channels, fpn_output_channels, kernel_size=3, stride=2, padding=1)

        self.p7_before = nn.ReLU()
        self.p7_output = Conv2d_dorefa(fpn_output_channels, fpn_output_channels, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        c3_feature, c4_feature, c5_feature = inputs
        p5_input_f = self.p5_input(c5_feature)
        p5_unsample_f = self.p5_unsample(p5_input_f)
        p5_output_f = self.p5_output(p5_input_f)

        p4_input_f = self.p4_input(c4_feature)
        p4_input_f = p5_unsample_f + p4_input_f
        p4_unsample_f = self.p4_unsample(p4_input_f)
        p4_output_f = self.p4_output(p4_input_f)

        p3_input_f = self.p3_input(c3_feature)
        p3_input_f = p4_unsample_f + p3_input_f
        p3_output_f = self.p5_output(p3_input_f)

        if self.use_p5:
            p6_output_f = self.p6_output(p5_output_f)
        else:
            p6_output_f = self.p6_output(c5_feature)

        del c3_feature, c4_feature, c5_feature
        p7_input_f = self.p7_before(p6_output_f)
        p7_output_f = self.p7_output(p7_input_f)

        return [p3_output_f, p4_output_f, p5_output_f, p6_output_f, p7_output_f]