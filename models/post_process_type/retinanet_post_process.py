import torch
import torch.nn as nn
from models.util import generate_predict_boxes, adjust_boxes
from torchvision.ops import nms
class post_process(nn.Module):
    def __init__(self):
        super(post_process, self).__init__()
        self.predict_boxes = generate_predict_boxes()
        self.adjust_box = adjust_boxes()
    def forward(self, input_batch, anchors, regressions, classifications):
        predict_boxes = self.predict_boxes(anchors, regressions)
        predict_boxes = self.adjust_box(predict_boxes, input_batch)
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