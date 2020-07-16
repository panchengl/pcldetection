from models.eval_type.csv_eval import evaluate
from models.eval_type.coco_eval import evaluate_coco
def evaluate_datasets(generator, retinanet, dataset_type, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None):
    if dataset_type == "csv":
        return evaluate(generator, retinanet, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None)
    elif dataset_type == "coco":
        return evaluate_coco(generator, retinanet)
    else:
        raise ValueError("dataset type error, please check dataset type")