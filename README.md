retinanet: first commit voc/coco map, input_size(608, 1024), letterbox_resize:

    voc:

        1.resnet_50             0.777

        2.resnet_101            0.792

        3.resnext_50_32x4d      0.793

        4.resnext_101_64x8d     0.825

        5.wide_resnet_50        0.801

        6.wide_resnet_101       *****

        7.resnet_152/18/34      *****

    coco:

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

