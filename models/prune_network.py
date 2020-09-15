import torch.nn as nn
import torch
from models.backbone import resnet, mobilenetv2, resnext, prune_resnet
from models.fpn_type import fpn, fcos_fpn
from models.head_type import retinanet_head, fcos_head
from models.loss_type import retinanet_loss, fcosnet_loss
from models.anchor_type import anchors
from models.post_process_type import retinanet_post_process, fcos_post_process
from models.post_process_type.fcos_post_process import DetectHead, ClipBoxes

model_factory = {
    # 'resnet_18':        resnet.resnet18,
    # 'resnet_34':        resnet.resnet34,
    # 'resnet_50':        resnet.resnet50,
    # 'resnet_101':       resnet.resnet101,
    # 'resnet_152':       resnet.resnet152,
    # "resnext_50":  resnext.resnext50_32x4d,
    # 'resnext_101': resnext.resnext101_32x8d,
    # 'wide_resnet_50':  resnext.wide_resnet50_2,
    # 'wide_resnet_101': resnext.wide_resnet101_2,
    # 'mobilenetv_2':     mobilenetv2.mobilenet_v2,
    'prune_resnet_50':  prune_resnet.prune_resnet_50,
    'prune_resnet_101':  prune_resnet.prune_resnet_101,
    'prune_resnet_152':  prune_resnet.prune_resnet_152,
}

fpn_factory = {
    "normal_fpn": fpn.FPN,
    "fcosnet_fpn": fcos_fpn.FPN,
}

head_factory = {
    "normal_head_class": retinanet_head.ClassificationHead,
    "normal_head_regression": retinanet_head.RegressionHead,
    "fcos_head": fcos_head.class_centerness_regression_head,
}

loss_factory = {
    "focal_loss":  retinanet_loss.FocalLoss,
    "fcosnet_loss": fcosnet_loss.fcosnet_loss
}

anchors_factory = {
    "normal_anchor": anchors.Anchors
}

post_precess_factory = {
    "normal_postprocess": retinanet_post_process.post_process,
    "fcosnet_postprocess": fcos_post_process.fcosnet_post_process,
}
class create_network_prune(nn.Module):
    def __init__(self, model_config, activate_channels_list, save_channels_list):
        super(create_network_prune, self).__init__()
        self.model_config = model_config
        self.is_training = False
        self.algorithm = self.model_config["type"]
        self.activate_channels_list = activate_channels_list
        self.save_channels_list = save_channels_list
        # self.net = net
        def build_backbone():
            backbone_type = "prune_" + self.model_config["backbone"]["type"] + '_' + self.model_config["backbone"]["depth"]
            backbone_module = model_factory[backbone_type](num_classes=self.model_config["dataset"]["num_classes"], activate_channels_list=self.activate_channels_list, save_channels_list=self.save_channels_list)
            return backbone_module

        def build_fpn():
            fpn_type = self.model_config["fpn"]["type"]
            fpn_module= fpn_factory[fpn_type](self.model_config["fpn"]["input_channels"])
            return fpn_module

        def build_head():
            head_type = self.model_config["head"]["type"]
            fpn_output_channels = self.model_config["head"]["output_channels"]
            num_classes = self.model_config["dataset"]["num_classes"]
            if head_type == "normal_head":
                num_anchors = len(self.model_config["anchor"]["anchor_scales"]) * len(self.model_config["anchor"]["anchor_rations"])
                regression_module = head_factory[head_type +'_regression'](fpn_output_channels=fpn_output_channels, num_anchors=num_anchors, regression_channels=fpn_output_channels)
                classification_module = head_factory[head_type + '_class'](fpn_output_channels=fpn_output_channels, num_anchors=num_anchors, num_classes=num_classes, classify_channels=fpn_output_channels)
                return classification_module, regression_module
            elif head_type == "fcos_head":
                class_centerness_reg_module = head_factory[head_type](fpn_output_channels=fpn_output_channels, class_num=num_classes, output_channels=fpn_output_channels,
                                                                      grounp_norm=self.model_config["head"]["grounp_norm"], prior=self.model_config["head"]["prior"])
                return class_centerness_reg_module
            else:
                raise ValueError("thie code version only support retinanet_head and fcosnet_head")

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
        if self.algorithm == "retinanet":
            self.ClassificationHead, self.RegressionHead = build_head()
            self.anchors = build_anchor()
            self.post_process = retinanet_post_process.post_process()
        elif self.algorithm == "fcosnet":
            self.fpn = build_fpn()
            self.class_centerness_reg_module = build_head()
            self.target_layer_2 = fcosnet_loss.generate_fcos_targets(strides=self.model_config["strides"], limit_range=self.model_config["limit_range"])
            self.detection_head = DetectHead(score_threshold=0.05, nms_iou_threshold=0.5, max_detection_boxes_num=1000, strides=self.model_config["strides"])
            self.clip_boxes = ClipBoxes()
        self.loss     = build_loss()

    def forward(self, inputs, is_training=False):
        """

        :type is_training: object
        """
        self.is_training = is_training
        if self.is_training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        features = self.backbone(img_batch)
        features = self.fpn(features)
        if self.algorithm == "retinanet":
            regressions = torch.cat([self.RegressionHead(feature) for feature in features], dim=1)
            classifications = torch.cat([self.ClassificationHead(feature) for feature in features], dim=1)
            anchors = self.anchors(img_batch)
            if self.is_training:
                return  self.loss(classifications, regressions, anchors, annotations, self.model_config["loss"]["positive_iou_thr"], self.model_config["loss"]["negative_iou_thr"])
            else:
                return self.post_process(input_batch=img_batch, anchors=anchors, regressions=regressions, classifications=classifications)
        elif self.algorithm == "fcosnet":
            classifications, centerness_preds, regressions = self.class_centerness_reg_module(features)
            out = [classifications, centerness_preds, regressions]
            add_centerness = self.model_config["add_centerness"]
            if self.is_training:
                batch_boxes = annotations[:, :, :-1]
                batch_labels = annotations[:, :, -1]
                targets = self.target_layer_2([out, batch_boxes, batch_labels])
                return self.loss([out, targets], add_centerness)
            else:
                scores, classes, boxes = self.detection_head(out)
                boxes = self.clip_boxes(img_batch, boxes)
                return  scores, classes, boxes

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

if __name__ == "__main__":
    from cfgs.retinanet_cfg import model_cfg
    from torch.autograd import Variable
    model = create_network_prune(model_cfg)
    scores, labels, boxes = model(Variable(torch.randn(8, 3,224,224)))
    print(scores.size())
    print(labels.size())
    print(boxes.size())
    # scores_grads = Variable( torch.randn(scores.size()), requires_grad=True)
    # labels_grads = Variable( torch.randn(labels.size()), requires_grad=True)
    # scores.backward(scores_grads, retain_graph=True)
    # labels.backward(labels_grads,retain_graph=True)