In this project, i want to reproduce some object detections,now, just make retinanet, fcos, there is map in coco dataset and voc dataset:

voc dataset: first commit voc/coco map,  letterbox_resize(u can easily get this result from official pretrained models):


    voc(input_size(608, 1024)): retinanet(epoch)        fcosnet(epoch)

        1.resnet_50             0.777(16)               0.763(18)

        2.resnet_101            0.792(19)               0.792(19)

        3.resnext_50_32x4d      0.793(16)               0.769(15)

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

fcosnet:

