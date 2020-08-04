In this project, i want to reproduce some object detections,now, just make retinanet, fcos, there is map in coco dataset and voc dataset:

voc dataset: first commit voc/coco map,  letterbox_resize(u can easily get this result from official pretrained models):


    voc(input_size(608, 1024)): retinanet(epoch)        fcosnet(epoch)        fcosnet(epoch-1333x800)

        1.resnet_50             0.777(16)               0.763(18)             0.773(11)

        2.resnet_101            0.792(19)               0.792(19)             0.803(14)

        3.resnext_50_32x4d      0.793(16)               0.769(15)             0.786(16)

        4.resnext_101_64x8d     0.825(10)               0.815(16)

        5.wide_resnet_50        0.801(26)               0.795(26)

        6.wide_resnet_101       *****                   *****

        7.resnet_152/18/34      *****                   *****


coco dataset:

    coco(resnet50-epoch30):

         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.481
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.128
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.430
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.484
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641

how to train your own dataset:

    1. create dataset, make your dataset become txt file: just like this:

        img_id img_dir width height label xmin ymin xmax ymax
        1 dir/img1.jpg width height label1 xmin ymin xmax ymax label2 xmin ymin xmax ymax .......
        2 dir/img2.jpg width height label1 xmin ymin xmax ymax .......

    2.create label.names just like data/voc.names in my code

    3.edit train/val file in cfg/net_config.py

    4.python main_fcos.py/main_retinanet.py
