import torch
import torch.nn as nn
import numpy as np

"""
fix the anchor with trained regression
"""
class generate_predict_boxes(nn.Module):

    def __init__(self, mean=None, std=None):
        super(generate_predict_boxes, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, anchors, regressions):

        widths  = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x   = anchors[:, :, 0] + 0.5 * widths
        ctr_y   = anchors[:, :, 1] + 0.5 * heights

        # dx = regressions[:, :, 0] * self.std[0] + self.mean[0]
        # dy = regressions[:, :, 1] * self.std[1]+ self.mean[1]
        # dw = regressions[:, :, 2] * self.std[2] + self.mean[2]
        # dh = regressions[:, :, 3] * self.std[3] + self.mean[3]

        dx = regressions[:, :, 0].cuda() * self.std[0].cuda() + self.mean[0].cuda()
        dy = regressions[:, :, 1].cuda() * self.std[1].cuda() + self.mean[1].cuda()
        dw = regressions[:, :, 2].cuda() * self.std[2].cuda() + self.mean[2].cuda()
        dh = regressions[:, :, 3].cuda() * self.std[3].cuda() + self.mean[3].cuda()

        predict_ctr_x = ctr_x + dx * widths
        predict_ctr_y = ctr_y + dy * heights
        predict_w     = torch.exp(dw) * widths
        predict_h     = torch.exp(dh) * heights

        predict_boxes_x1 = predict_ctr_x - 0.5 * predict_w
        predict_boxes_y1 = predict_ctr_y - 0.5 * predict_h
        predict_boxes_x2 = predict_ctr_x + 0.5 * predict_w
        predict_boxes_y2 = predict_ctr_y + 0.5 * predict_h

        predict_boxes = torch.stack([predict_boxes_x1, predict_boxes_y1, predict_boxes_x2, predict_boxes_y2], dim=2)

        return predict_boxes


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std
        print("self.std is", self.std)
        print("self.mean is", self.mean)
    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0].cuda() + self.mean[0].cuda()
        dy = deltas[:, :, 1] * self.std[1].cuda() + self.mean[1].cuda()
        dw = deltas[:, :, 2] * self.std[2].cuda() + self.mean[2].cuda()
        dh = deltas[:, :, 3] * self.std[3].cuda() + self.mean[3].cuda()

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


"""
adjust the predict box to the image
"""
class adjust_boxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(adjust_boxes, self).__init__()

    def forward(self, boxes, image):
        batch_size, num_channels, height, width = image.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std
        print("self.std is", self.std)
        print("self.mean is", self.mean)

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

def load_model(model, optimizer, model_dir):
    checkpoint_dict = torch.load(model_dir)
    model.load_state_dict(checkpoint_dict["net"], strict=True)
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    current_epoch = checkpoint_dict['epoch']
    net_name = str(model_dir).split("/")[-1].split("_")[4] +  '_' + str(model_dir).split("/")[-1].split("_")[5]
    print("finished restore model from {}, backbone is {}, retrain from epoch {}".format(model_dir, net_name, str(current_epoch)))
    return model, optimizer, current_epoch

def create_datastes(cfg, **kwargs):
    from models.dataloader_type.dataloader import CSVDataset, Resizer, Augmenter, Normalizer, CocoDataset
    from torchvision import transforms
    min_side, max_side = cfg['input_size']
    dataset_type, input_size = cfg["dataset"]["type"], cfg["input_size"]
    mean, std = cfg["mean"], cfg["std"]
    if dataset_type == 'coco':
        try:
            coco_path = cfg["dataset"]["coco_path"]
        except:
            raise ValueError("coco dataset must provide coco dir")
        if coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif dataset_type == 'csv':
        try:
            csv_train, csv_val, csv_classes = cfg["dataset"]["csv_train"], cfg["dataset"]["csv_val"], cfg["dataset"]["csv_classes"]
            add_background = cfg["dataset"]["add_background"]
        except:
            raise ValueError("csv dataset must provide csv dir names")
        dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                                   transform=transforms.Compose([Normalizer(mean=mean, std=std), Augmenter(), Resizer(min_side=min_side, max_side=max_side)]), add_backbround=add_background)
        dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes,transform=transforms.Compose([Normalizer(mean=mean, std=std), Resizer(min_side=min_side, max_side=max_side)]))# be careful, val dataset must use add_background=False
        if add_background:
            print("be careful, this algorithm use background as label 0 when model is training , class_id becomes class_num+1, just like label0 -> {}".format(1))
            print("test model will auto change back")
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    return dataset_train, dataset_val