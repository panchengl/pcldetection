import torch
import numpy as np
import torch.nn as nn

from models.post_process_type import retinanet_post_process
def distillation_loss2(data, anchors, output_s, output_t, cfg):
    reg_m = 0.0
    T = 3.0
    # Lambda_cls, Lambda_box = 0.0001, 0.001
    Lambda_cls, Lambda_box = 0.01, 0.1
    regressions_s, classicifications_s = output_s
    regressions_t, classicifications_t = output_t
    img_batch = data['img'].cuda().float()
    anno = data['annot'].cuda()
    from models.loss_type import retinanet_loss
    loss_fun = retinanet_loss.FocalLoss_reg()
    _, regression_loss_s = loss_fun(classicifications_s, regressions_s, anchors, data['annot'].cuda(), cfg["loss"]["positive_iou_thr"], cfg["loss"]["negative_iou_thr"])
    _, regression_loss_t = loss_fun(classicifications_t, regressions_t, anchors, data['annot'].cuda(), cfg["loss"]["positive_iou_thr"], cfg["loss"]["negative_iou_thr"])
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    ft = torch.cuda.FloatTensor if output_s[0].is_cuda else torch.Tensor
    lcls, lbox = ft([0]), ft([0])
    reg_ratio, reg_num, reg_nb = 0, 0, 0
    l2_num = regression_loss_s > regression_loss_t
    # print("regression_loss_s shape is", regression_loss_s.shape)
    # print("regression_loss_tshape is ", regression_loss_t.shape)
    # print("t reg loss is", regression_loss_t)
    # print("s reg loss is", regression_loss_s)
    # print("l2 num is", l2_num)
    # print("regression_loss_t[l2_num] is", regression_loss_s[l2_num])
    lbox += regression_loss_t[l2_num].sum()
    reg_num += l2_num.sum().item()
    reg_nb += regression_loss_s.shape[0]
    # print("reg_nb is", reg_nb)
    lcls += criterion_st(nn.functional.log_softmax(classicifications_s.view(-1, 4) / T, dim=1),
                         nn.functional.softmax(classicifications_t.view(-1, 4) / T, dim=1)) * (T * T) / reg_nb  # batch_size
    if reg_nb:
        reg_ratio = reg_num / reg_nb
    # print("reg_num is", reg_num)
    # print("reg_nb is", reg_nb)
    # print("reg_ration is", reg_ratio)
    # print("cls is", lcls)
    return lcls * Lambda_cls + lbox * Lambda_box, reg_ratio


def distillation_loss1(regressions_s, classicifications_s, regressions_t, classicifications_t, num_classes, batch_size):
    T = 3.0
    Lambda_ST = 0.01
    # Lambda_ST = 1
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    output_s = torch.cat([regressions_s.view(-1, 4), classicifications_s.view(-1, num_classes)],  dim=-1)
    output_t = torch.cat([regressions_t.view(-1, 4), classicifications_t.view(-1, num_classes)],  dim=-1)
    loss_st = criterion_st(torch.nn.functional.log_softmax(output_s / T, dim=1),
                           torch.nn.functional.softmax(output_t / T, dim=1)) * (T * T) / batch_size
    # print("loss_st * Lambda_ST is", loss_st )
    return loss_st * Lambda_ST
