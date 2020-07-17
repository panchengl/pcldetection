import torch.nn as nn
import torch
from models.backbone import resnet, mobilenetv2, resnext
from models.fpn_type import fpn
from models.head_type import normal_head
from models.loss_type import focal_losses
from models.anchor_type import anchors
from models.post_process_type import normal_post_process

model_factory = {
    'resnet_18':        resnet.resnet18,
    'resnet_34':        resnet.resnet34,
    'resnet_50':        resnet.resnet50,
    'resnet_101':       resnet.resnet101,
    'resnet_152':       resnet.resnet152,
    "resnext_50":  resnext.resnext50_32x4d,
    'resnext_101': resnext.resnext101_32x8d,
    'wide_resnet_50':  resnext.wide_resnet50_2,
    'wide_resnet_101': resnext.wide_resnet101_2,
    'mobilenetv_2':     mobilenetv2.mobilenet_v2,
}

fpn_factory = {
    "normal_fpn": fpn.FPN,
}

head_factory = {
    "normal_head_class": normal_head.ClassificationHead,
    "normal_head_regression": normal_head.RegressionHead,
}

loss_factory = {
    "focal_loss":  focal_losses.FocalLoss,
}

anchors_factory = {
    "normal_anchor": anchors.Anchors
}

class create_network(nn.Module):
    def __init__(self, model_config):
        super(create_network, self).__init__()
        self.model_config = model_config
        self.is_training = False
        def build_backbone():
            backbone_type = self.model_config["backbone"]["type"] + '_' + self.model_config["backbone"]["depth"]
            backbone_module = model_factory[backbone_type](num_classes=self.model_config["dataset"]["num_classes"], pretrained=True)
            return backbone_module

        def build_fpn():
            fpn_type = self.model_config["fpn"]["type"]
            fpn_module= fpn_factory[fpn_type](self.model_config["fpn"]["input_channels"])
            return fpn_module

        def build_head():
            head_type = self.model_config["head"]["type"]
            num_anchors = len(self.model_config["anchor"]["anchor_scales"]) * len(self.model_config["anchor"]["anchor_rations"])
            fpn_output_channels =  self.model_config["head"]["output_channels"]
            num_classes = self.model_config["dataset"]["num_classes"]
            regression_module = head_factory[head_type +'_regression'](fpn_output_channels=fpn_output_channels, num_anchors=num_anchors, regression_channels=fpn_output_channels)
            classification_module = head_factory[head_type + '_class'](fpn_output_channels=fpn_output_channels, num_anchors=num_anchors, num_classes=num_classes, classify_channels=fpn_output_channels)
            return classification_module, regression_module

        def build_loss():
            loss_type = self.model_config["loss"]["type"]
            loss_module = loss_factory[loss_type]()
            return loss_module

        def build_anchor():
            anchor_type = self.model_config["anchor"]["type"]
            pyramid_levels = self.model_config["anchor"]["pyramid_levels"]
            anchor_ratios = self.model_config["anchor"]["anchor_rations"]
            anchor_scales = self.model_config["anchor"]["anchor_scales"]
            anchor_module = anchors_factory[anchor_type](pyramid_levels=pyramid_levels, ratios=anchor_ratios, scales=anchor_scales)
            return anchor_module

        self.backbone = build_backbone()
        self.fpn      = build_fpn()
        self.ClassificationHead, self.RegressionHead = build_head()
        self.loss     = build_loss()
        self.anchors   = build_anchor()
        self.post_process = normal_post_process.post_process()


    def forward(self, inputs, is_training=False):
        self.is_training = is_training
        if self.is_training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        features = self.backbone(img_batch)
        features = self.fpn(features)
        regressions = torch.cat([self.RegressionHead(feature) for feature in features], dim=1)
        classifications = torch.cat([self.ClassificationHead(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)
        if self.is_training:
            return  self.loss(classifications, regressions, anchors, annotations, self.model_config["loss"]["positive_iou_thr"], self.model_config["loss"]["negative_iou_thr"])
        else:
            return self.post_process(input_batch=img_batch, anchors=anchors, regressions=regressions, classifications=classifications)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

if __name__ == "__main__":
    from cfgs.retinanet_cfg import model_cfg
    from torch.autograd import Variable
    model = create_network(model_cfg, is_training=False)
    scores, labels, boxes = model(Variable(torch.randn(8, 3,224,224)))
    print(scores.size())
    print(labels.size())
    print(boxes.size())
    # scores_grads = Variable( torch.randn(scores.size()), requires_grad=True)
    # labels_grads = Variable( torch.randn(labels.size()), requires_grad=True)
    # scores.backward(scores_grads, retain_graph=True)
    # labels.backward(labels_grads,retain_graph=True)