from torch import nn
import torch
from torch.autograd import Variable
from models.fpn_type.fpn import FPN
from torchvision.ops import nms

from models.head_type.normal_head import ClassificationHead, RegressionHead, ClassificationModel_variety_input, RegressionModel_variety_input
from models.anchor_type.anchors import Anchors
from models.loss_type.focal_losses import FocalLoss
from models.util import generate_predict_boxes, adjust_boxes

import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math

__all__ = ['MobileNetV2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, width_mult=1.0, round_nearest=8, **kwargs):
        super(MobileNetV2, self).__init__()
        print("this model is begining init mobilenetv2 backbone, so network depth params is invalid")
        self.num_classes = num_classes
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 0
            [6, 24, 2, 2],  # 1
            [6, 32, 3, 2],  # 2
            [6, 64, 4, 2],  # 3
            [6, 96, 3, 1],  # 4
            [6, 160, 3, 2],  # 5
            [6, 320, 1, 1],  # 6
        ]
        self.feat_id = [1, 2, 4, 6]
        self.feat_channel = []
        print("kwargs is", kwargs)
        try:
            self.anchor_ratios = kwargs['anchor_rations']
            self.anchor_scales = kwargs['anchor_scales']
            self.pyramid_levels = kwargs['pyramid_levels']
            self.positive_iou_thr = kwargs["positive_iou_thr"]
            self.negative_iou_thr = kwargs["negative_iou_thr"]
        except:
            print("anchor params is fixed, this model may only use inference")
            self.anchor_ratios = [0.5, 1, 2]
            self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            self.pyramid_levels = [3, 4, 5, 6, 7]
            self.positive_iou_thr = 0.5
            self.negative_iou_thr = 0.4

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id:
                self.__setattr__("feature_%d" % id, nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                print("id is", id)
                print("output_channel is ", output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.fpn = FPN(self.feat_channel[1], self.feat_channel[2], self.feat_channel[3])
        # self.RegressionHead = RegressionHead(fpn_output_channels=256, num_anchors=self.num_anchors, regression_channels=256)
        # self.ClassificationHead = ClassificationHead(fpn_output_channels=256, num_anchors=self.num_anchors, num_classes=self.num_classes,
        #                                              classify_channels=256)
        self.RegressionHead = RegressionModel_variety_input(num_features_in=320)
        self.ClassificationHead = ClassificationModel_variety_input(num_features_in=320,num_anchors=self.num_anchors, num_classes=self.num_classes)
        self.anchors = Anchors(pyramid_levels=self.pyramid_levels, ratios=self.anchor_ratios, scales=self.anchor_scales)
        self.predict_boxes = generate_predict_boxes()
        self.adjust_box = adjust_boxes()
        self.focalLoss = FocalLoss()

    def forward(self, x,  is_training=False):
        y = []
        # print("is_training : ", is_training)
        if is_training:
            img_batch, annotations = x
        else:
            img_batch = x
        x_input_ = img_batch
        for id in self.feat_id:
            x_input_ = self.__getattr__("feature_%d" % id)(x_input_)
            y.append(x_input_)
        # features = self.fpn( [y[1], y[2], y[3]] )
        features = y[3]
        regressions = torch.cat([self.RegressionHead(feature) for feature in features], dim=1)
        classifications = torch.cat([self.ClassificationHead(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)
        if self.training:
            return self.focalLoss(classifications, regressions, anchors, annotations, self.positive_iou_thr,
                                  self.negative_iou_thr)
        else:
            predict_boxes = self.predict_boxes(anchors, regressions)
            predict_boxes = self.adjust_box(predict_boxes, img_batch)

            # finalResult = [[], [], []]
            pred_scores = torch.Tensor([]).cuda()
            pred_labels = torch.Tensor([]).long().cuda()
            finalAnchorBoxesCoordinates = torch.Tensor([]).cuda()
            for i in range(classifications.shape[2]):
                scores = torch.squeeze(classifications[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(predict_boxes)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                pred_scores = torch.cat((pred_scores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                pred_labels = torch.cat((pred_labels, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [pred_scores, pred_labels, finalAnchorBoxesCoordinates]

        # return feature_last

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def dict2list(func):
    def wrap(*args, **kwargs):
        self = args[0]
        x = args[1]
        ret_list = []
        ret = func(self, x)
        for k, v in ret[0].items():
            ret_list.append(v)
        return ret_list

    return wrap


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def mobilenet_v2(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(num_classes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2'], model_dir='.'), strict=False)
    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    mobilenet = MobileNetV2(num_classes=5)
    pred = mobilenet(Variable(torch.randn(2, 3, 224, 224)))
    for i in range(len(pred)):
        print("pred_{}_shape is {}".format(i, pred[i].shape))
    # print(loc_preds.size())
    # print(cls_preds.size())
    # loc_grads = Variable(torch.randn(loc_preds.size()))
    # cls_grads = Variable(torch.randn(cls_preds.size()))
    # loc_preds.backward(loc_grads, retain_graph=True)
    # cls_preds.backward(cls_grads, retain_graph=True)

