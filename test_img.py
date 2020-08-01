import torch
import numpy as np
import time
import os
import cv2
from models.network import create_network

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1
        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result

def load_classes_names(name_reader):
    result = {}
    lines = name_reader.readlines()
    for class_id, line in enumerate(lines):
        label = line.strip()
        result[label] = class_id
    return result
# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(model_cfg, inference_cfg):
    with open(model_cfg["dataset"]["csv_classes"], 'r') as f:
        classes = load_classes_names(f)
    labels = {}
    for key, value in classes.items():
        labels[value] = key
    model = create_network(model_cfg)
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint_dict = torch.load(inference_cfg["test_model_path"])
    print("checkpoint is ", checkpoint_dict['net'].keys())
    model.load_state_dict(checkpoint_dict["net"], strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.training = False
    model.eval()
    total_time = 0
    for img_name in os.listdir(inference_cfg["test_img_dir"]):

        image = cv2.imread(os.path.join(inference_cfg["test_img_dir"], img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = inference_cfg["input_size"][0]
        max_side = inference_cfg["input_size"][1]
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            scores = torch.clamp(scores, 1e-4, 1.0 - 1e-4)
            # print("scores is ", scores)
            # print("classification is ", classification)
            # print("box is ", transformed_anchors.shape)
            print('Elapsed time: {}'.format(time.time() - st))
            total_time += (time.time() - st)
            idxs = np.where(scores.cpu() > inference_cfg["conf"])
            # print("idx is", idxs)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                # print(bbox, classification.shape, scores[j], classification[idxs[0][j]])
                score = scores[idxs[0][j]]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.namedWindow("detections", 0)
            cv2.resizeWindow("detections", 1200, 800)
            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    average_time = total_time/len(os.listdir(inference_cfg["test_img_dir"]))
    print("average_time is ", average_time)

if __name__ == '__main__':

    # from cfgs.retinanet_cfg import model_cfg, inference_cfg
    # from cfgs.retinanet_resnet_wide_cfg import model_cfg, inference_cfg
    # from cfgs.retinanet_resneXt_cfg import model_cfg, inference_cfg
    from cfgs.fcosnet_cfg import model_cfg, inference_cfg
    detect_image(model_cfg, inference_cfg)
