# model settings
model_cfg = dict(
    type='retinanet',
    input_size=[608, 1024],
    use_gpus="4,5,6,7",
    batch_size=24,
    num_workers=3,
    lr = 1e-5,
    save_dir = '/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/prune_checkpoint',
    restore_path = '/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/csv_retinanet_epoch_14_resnet_50_map_0.774_lr_1e-05.pt',
    epochs = 100,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    dataset=dict(
        type="csv",   # if type=csv, cfg will support train_file, val_file, names
        coco_path="/home/pcl/data/coco",
        csv_train = '/home/pcl/pytorch_work/my_github/pcldetection/data/train.txt',
        csv_val = '/home/pcl/pytorch_work/my_github/pcldetection/data/val.txt',
        csv_classes = '/home/pcl/pytorch_work/my_github/pcldetection/data/voc.names',
        num_classes=20,
        add_background=False),
    backbone=dict(
        type='resnet',
        depth='50',
        block="bottlenect",
        prune=True,
        prune_ration=0.2,
        prune_type="sliming",
        sliming_epochs=100,
        gamma=1e-4,
        sliming_save_dir="/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/sliming_checkpoint/",
        sliming_path="/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/sliming_checkpoint/csv_model_epoch_13_resnet_50_map_0.547_lr_0.0001.pt",  # l1-norm, sliming, ...
        prune_save_path = "/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/prune_checkpoint/",
        prune_path=""),
        # restore_path=""),  # l1-norm, sliming, ...
    fpn=dict(
        type='normal_fpn',
        input_channels=[512, 1024, 2048]),
    head=dict(
        type='normal_head',
        output_channels=256),
    loss=dict(
        type='focal_loss',
        positive_iou_thr=0.5,
        negative_iou_thr = 0.4),
    anchor=dict(
        type='normal_anchor',
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        anchor_rations = [0.5, 1, 2],
        pyramid_levels = [3, 4, 5, 6, 7]
    ),
    nms_thr = 0.5,
    mean = [0.485, 0.456, 0.406],      # mean is not effect, fixed this value in code
    std = [0.229, 0.224, 0.225],
)
inference_cfg = dict(
    test_img_dir = "img_dir/voc",
    test_model_path = '/home/pcl/pytorch_work/my_github/pcldetection/checkpoint/csv_retinanet_epoch_10_resnext_101_map_0.82_lr_1e-05.pt',
    conf = 0.3,
)

print("model is", model_cfg)
print("model type is ", model_cfg['type'])
print("model backbone is", model_cfg['backbone']['type'])
print("model backbone depth is", model_cfg['backbone']['depth'])