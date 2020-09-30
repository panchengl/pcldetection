import collections
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.dataloader_type.dataloader import collater, AspectRatioBasedSampler
from models.util import load_model, create_datastes, load_model_sigle_gpu
from models.network import create_network
from models.network_knowledge_distill import create_network_no_anchor
from models.prune_type import prune_sliming
from models.prune_network_knowledge_distill import create_network_prune
from utils.eval_util import evaluate_datasets
# from cfgs.fcosnet_cfg import model_cfg as cfg
from cfgs.retinanet_prune_cfg import model_cfg as cfg
from models.backbone import prune_resnet
assert torch.__version__.split('.')[0] == '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = cfg["use_gpus"]
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
def main():
    dataset_train, dataset_val = create_datastes(cfg)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=cfg["batch_size"], drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=cfg["num_workers"], collate_fn=collater, batch_sampler=sampler)
    # Create the model
    print("dataset_train.num_classes() is ", dataset_train.num_classes())

    ######### first stage: create sliming network ########
    if int(cfg["backbone"]["depth"]) in [18, 34, 50, 101, 152]:
        model = create_network(cfg)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    assert cfg["restore_path"] is not None
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    optimizer_ori = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ori, patience=3, verbose=True)
    if len(cfg["backbone"]["sliming_path"]) != 0:
        model, _, _ = load_model(model, optimizer_ori, cfg["backbone"]["sliming_path"])
    ########################################################

    ######### second stage: create prune net and copy weights ########
    print("wariming:  this version only support prune backbone, if u want prune whole network, u can add issues with me")
    activate_channels_list, save_channels_list = prune_resnet.get_activate_channel_sliming(model, global_percent=cfg["backbone"]["prune_ration"])
    prune_net = create_network_prune(cfg, activate_channels_list, save_channels_list)
    prune_net = prune_resnet.pruning(prune_net, model)
    del model
    # print(prune_net)
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        prune_net = torch.nn.DataParallel(prune_net).cuda()
    else:
        prune_net = torch.nn.DataParallel(prune_net)
    optimizer = optim.Adam(prune_net.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    prune_net.train()
    ###################################################################

    ######### third stage: create teacher net  ########################
    model_t = create_network_no_anchor(cfg)
    optimizer_t = optim.Adam(model_t.parameters(), lr=cfg["lr"])
    assert cfg["restore_path"] != None
    model_t, _, _ = load_model(model_t, optimizer_t, cfg["restore_path"])
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        model_t = torch.nn.DataParallel(model_t).cuda()
    else:
        model_t = torch.nn.DataParallel(model_t)
    ###################################################################

    # prune_net.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))
    optimizer.param_groups[0]['lr'] = 1.25e-4
    best_ap = 1e-5
    # if len(cfg["restore_path"]) != 0:
    start_epoch = 0
    for epoch_num in range(start_epoch, cfg["epochs"]):
        prune_net.train()
        prune_net.is_training = True
        epoch_loss = []
        batch_num = (len(dataloader_train))
        print("begining current batch training, current training batch_num is %s"%(str(int(batch_num))))
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            regressions_s, classicifications_s = prune_net([data['img'].cuda().float(), data['annot']], is_training=True)
            regressions_t, classicifications_t = model_t([data['img'].cuda().float(), data['annot']], is_training=True)
            from models.anchor_type import anchors
            pyramid_levels = cfg["anchor"]["pyramid_levels"]
            anchor_ratios = cfg["anchor"]["anchor_rations"]
            anchor_scales = cfg["anchor"]["anchor_scales"]
            anchors = anchors.Anchors( pyramid_levels=pyramid_levels, ratios=anchor_ratios, scales=anchor_scales)
            calc_anchors = anchors(data['img'].cuda().float())
            # print("anchors shape is", calc_anchors.shape)
            from models.loss_type import retinanet_loss
            loss_fun = retinanet_loss.FocalLoss()
            classification_loss, regression_loss = loss_fun(classicifications_s, regressions_s, calc_anchors, data['annot'].cuda(), cfg["loss"]["positive_iou_thr"], cfg["loss"]["negative_iou_thr"])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            from models.loss_type import knowledge_distill_loss
            # soft_target = knowledge_distill_loss.distillation_loss1(regressions_s, classicifications_s, regressions_t, classicifications_t, cfg["dataset"]["num_classes"], cfg["batch_size"])
            soft_target, reg_ratio = knowledge_distill_loss.distillation_loss2(data, anchors=calc_anchors, output_s=(regressions_s, classicifications_s),
                                                                               output_t=(regressions_t, classicifications_t), cfg=cfg)
            loss = classification_loss + regression_loss + soft_target
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prune_net.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            if iter_num % 100 == 0:
                print(
                    'Runing lr: {} | Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | soft_target loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        float(optimizer.param_groups[0]["lr"]), epoch_num, iter_num, float(classification_loss),
                        float(regression_loss), float(soft_target), np.mean(loss_hist)))
            del classification_loss
            del regression_loss
            del soft_target
        scheduler.step(np.mean(epoch_loss))
        print('Evaluating dataset')
        prune_net.eval()
        if epoch_num %5 == 0 or epoch_num > 10:
            mAP = evaluate_datasets(dataset_val, prune_net, cfg["dataset"]["type"], iou_threshold=cfg["iou_threshold"], score_threshold=cfg["score_threshold"], max_detections=cfg["max_detections"])
            if mAP != None:
                if mAP["mean_ap"] > best_ap:
                    best_ap = mAP["mean_ap"]
                    checkpoint = {"net":prune_net.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch_num}
                    torch.save(checkpoint, os.path.join(cfg["save_dir"], 'prune_{}_{}_model_epoch_{}_{}_map_{}_lr_{}.pt'.format(
                    cfg["dataset"]["type"], cfg["type"], epoch_num, (cfg["backbone"]["type"] + '_' + cfg["backbone"]["depth"]), str(round(mAP['mean_ap'], 3)),  str(float(optimizer.param_groups[0]["lr"])) ) ) )
            else:
                print("The coco calculation standard uses its own parameters,just like conf=0.05, iou=[0.1.....0.9..]")
                torch.save(prune_net.module, '{}_model_{}.pt'.format(cfg["dataset"]["type"], epoch_num))
            # scheduler.step(mAP["mean_ap"])
        prune_net.eval()
        torch.save(prune_net, 'model_final.pt')

if __name__ == '__main__':
    main()
