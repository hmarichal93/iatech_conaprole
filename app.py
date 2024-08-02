#import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import supervision as sv
import yolov5

from pathlib import Path
class Pipeline:
    def __init__(self, model_path='/data/ia_tech_conaprole/repos/Dense-Object-Detection/weights/best.pt',
                    output_dir="./results",
                    device=None,
                    matcher=None,
                    debug=True):
        self.model = self.load_yolov5_model(model_path)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def load_yolov5_model(self, model_path):
        model = yolov5.load(model_path)
        # set model parameters
        model.conf = 0.25  # NMS confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        model.max_det = 1000  # maximum number of detections per image
        model.size = 640  # image size
        return model

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resize image
        #img = cv2.resize(img, (640, 640))
        return img

    def draw_bounding_boxes(self, image, boxes, color = (255, 0, 0), thickness = 3):
        for box in boxes:
            x1, y1, x2, y2 = box
            #convert coordenates to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        return image

    def write_image(self, image):
        output_path = f"results/detection_{self.output_prefix}_nms_{self.model.conf}_iou_{self.model.iou}_size_{self.model.size}.png"
        image_r = cv2.resize(image, (640, 640))
        image_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_r)

    def yolov5_inference(self, img):
        results = self.model(img)
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # show detection bounding boxes on image
        #results.show()
        if self.debug:
            image = self.draw_bounding_boxes(img.copy(), boxes)
            self.write_image(image)

        return boxes

    def classifier(self, image, boxes):

        pass

    def compute_metrics(self):
        pass

    def print_metrics(self, res):
        pass

    def main(self, image_path):
        self.output_prefix = Path(image_path).stem
        image = self.preprocess_image(image_path)
        boxes = self.yolov5_inference(image)
        res = self.classifier(image, boxes)
        res = self.compute_metrics()
        self.print_metrics(res)
        return res


def main(image_path):

    pipeline = Pipeline()
    pipeline.main(image_path)

    return

if __name__ == "__main__":
    image_path = "./assets/WhatsApp Image 2024-05-24 at 12.00.13 (2).jpeg"
    image_path = "./assets/IMG_9149.png"
    image_path = "./assets/IMG_9156.png"
    main(image_path)