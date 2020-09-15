# model settings
model_cfg = dict(
    type='fcosnet',
    input_size=[608, 1024],
    use_gpus="4,5,6,7",
    batch_size=16,
    num_workers=3,
    lr = 1e-3,
    save_dir = 'checkpoint/',
    restore_path = '',
    epochs = 100,
    iou_threshold=0.5,
    score_threshold=0.5,
    max_detections=1000,
    prior=0.01,
    strides=[8,16,32,64,128],
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]],
    use_GN_head=True,
    add_centerness = True,
    cnt_on_reg = True,
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    dataset=dict(
        type="csv",   # if type=csv, cfg will support train_file, val_file, names
        coco_path="/home/pcl/data/coco",
        csv_train = 'data/train.txt',
        csv_val = 'data/val.txt',
        csv_classes = './data/voc.names',
        num_classes=20,
        add_background=True),
    backbone=dict(
        type='resnet',
        depth="50",
        block="bottlenect",
        prune=True,
        prune_ration=0.2),
    fpn=dict(
        type='fcosnet_fpn',
        input_channels=[512, 1024, 2048]), #if not use p5
    head=dict(
        type='fcos_head',
        output_channels=256,
        grounp_norm=True,
        prior=0.01),
    loss=dict(
        type='fcosnet_loss',        # iou_loss or giou loss
        positive_iou_thr=0.5,
        negative_iou_thr = 0.4),
    nms_thr = 0.5,
)
inference_cfg = dict(
    test_img_dir = "img_dir/voc",
    test_model_path = 'checkpoint/csv_fcosnet_model_epoch_16_resnext_101_map_0.815_lr_1e-05.pt',
    conf = 0.3,
    input_size=[800, 1333],
)

print("model is", model_cfg)
print("model type is ", model_cfg['type'])
print("model backbone is", model_cfg['backbone']['type'])
print("model backbone depth is", model_cfg['backbone']['depth'])