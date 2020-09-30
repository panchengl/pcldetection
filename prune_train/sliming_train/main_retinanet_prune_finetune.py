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
from models.prune_type import prune_sliming
from models.prune_network import create_network_prune
from utils.eval_util import evaluate_datasets
# from cfgs.fcosnet_cfg import model_cfg as cfg
from cfgs.retinanet_prune_cfg import model_cfg as cfg
from models.backbone import prune_resnet
assert torch.__version__.split('.')[0] == '1'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["use_gpus"]
def main():
    dataset_train, dataset_val = create_datastes(cfg)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=cfg["batch_size"], drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=cfg["num_workers"], collate_fn=collater, batch_sampler=sampler)
    # Create the model
    print("dataset_train.num_classes() is ", dataset_train.num_classes())
    if int(cfg["backbone"]["depth"]) in [18, 34, 50, 101, 152]:
        model = create_network(cfg)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    assert cfg["restore_path"] is not None
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, verbose=True)
    # start_epoch = 0
    if len(cfg["restore_path"]) != 0:
        model, optimizer, start_epoch = load_model(model, optimizer, cfg["backbone"]["sliming_path"])
    print("wariming:  this version only support prune backbone, if u want prune whole network, u can add issues with me")
    activate_channels_list, save_channels_list = prune_resnet.get_activate_channel_sliming(model, global_percent=cfg["backbone"]["prune_ration"])
    prune_net = create_network_prune(cfg, activate_channels_list, save_channels_list)
    prune_net = prune_resnet.pruning(prune_net, model)
    # print(prune_net)
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        prune_net = torch.nn.DataParallel(prune_net).cuda()
    else:
        prune_net = torch.nn.DataParallel(prune_net)
    del model
    optimizer = optim.Adam(prune_net.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    if len(cfg["backbone"]["prune_path"]) != 0:
        model, optimizer, start_epoch = load_model(prune_net, optimizer, cfg["backbone"]["prune_path"])
    prune_net.train()
    # prune_net.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))
    optimizer.param_groups[0]['lr'] = 1.25e-4
    best_ap = 1e-5
    for epoch_num in range(start_epoch+1, cfg["epochs"]):
        prune_net.train()
        prune_net.is_training = True
        epoch_loss = []
        batch_num = (len(dataloader_train))
        print("begining current batch training, current training batch_num is %s"%(str(int(batch_num))))
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, regression_loss = prune_net([data['img'].cuda().float(), data['annot']],
                                                             is_training=True)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prune_net.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            if iter_num % 100 == 0:
                print(
                    'Runing lr: {} | Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        float(optimizer.param_groups[0]["lr"]), epoch_num, iter_num, float(classification_loss),
                        float(regression_loss), np.mean(loss_hist)))
            del classification_loss
            del regression_loss
        scheduler.step(np.mean(epoch_loss))
        print('Evaluating dataset')
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
