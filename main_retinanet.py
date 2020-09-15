import collections
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dataloader_type.dataloader import collater, AspectRatioBasedSampler
from models.util import load_model, create_datastes
from utils.eval_util import evaluate_datasets
from models.network import create_network
from cfgs.retinanet_cfg import model_cfg as cfg

assert torch.__version__.split('.')[0] == '1'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["use_gpus"]
def main():
    dataset_train, dataset_val = create_datastes(cfg)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=cfg["batch_size"], drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=cfg["num_workers"], collate_fn=collater, batch_sampler=sampler)
    # Create the model
    print("dataset_train.num_classes() is ", dataset_train.num_classes())
    if int(cfg["backbone"]["depth"]) in [18, 34, 50, 101, 152]:
        retinanet = create_network(cfg)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet = retinanet.cuda()
    if len(cfg["use_gpus"].replace(" ",'')) > 1:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)
    optimizer = optim.Adam(retinanet.parameters(), lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, verbose=True)
    start_epoch = 0
    if len(cfg["restore_path"]) != 0:
        retinanet, optimizer, start_epoch = load_model(retinanet, optimizer, cfg["restore_path"])
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()
    print('Num training images: {}'.format(len(dataset_train)))
    best_ap = 1e-5
    for epoch_num in range(start_epoch, cfg["epochs"]):
        retinanet.train()
        retinanet.is_training = True
        retinanet.module.freeze_bn()
        epoch_loss = []
        batch_num = (len(dataloader_train))
        print("begining current batch training, current training batch_num is %s"%(str(int(batch_num))))
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']], is_training=True)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            if iter_num%100 == 0:
                print('Runing lr: {} | Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}' .format(
                    float(optimizer.param_groups[0]["lr"]),epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)) )
            del classification_loss
            del regression_loss
        scheduler.step(np.mean(epoch_loss))
        print('Evaluating dataset')
        if epoch_num >= 10:
            mAP = evaluate_datasets(dataset_val, retinanet, cfg["dataset"]["type"], iou_threshold=cfg["iou_threshold"], score_threshold=cfg["score_threshold"], max_detections=cfg["max_detections"])
            if mAP != None:
                if mAP["mean_ap"] > best_ap:
                    best_ap = mAP["mean_ap"]
                    checkpoint = {"net":retinanet.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch_num}
                    torch.save(checkpoint, os.path.join(cfg["save_dir"], '{}_retinanet_epoch_{}_{}_map_{}_lr_{}.pt'.format(
                    cfg["dataset"]["type"], epoch_num, (cfg["backbone"]["type"] + '_' + cfg["backbone"]["depth"]), str(round(mAP['mean_ap'], 3)),  str(float(optimizer.param_groups[0]["lr"])) ) ) )
            else:
                print("The coco calculation standard uses its own parameters,just like conf=0.05, iou=[0.1.....0.9..]")
                torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(cfg["dataset"]["type"], epoch_num))
            # scheduler.step(mAP["mean_ap"])
            retinanet.eval()
    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
