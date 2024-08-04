import cv2
#import yolov5

from pathlib import Path
import numpy as np
import torch
from weak_labelling import matcher
from classifier import Device, Loftr_Classifier, Sift_Classifier
from feature_matching import resize_image_using_pil_lib
from PIL import Image

from dob.yolov5.models.experimental import attempt_load
from dob.yolov5.utils.general import non_max_suppression
from dob.yolov5.utils.torch_utils import select_device

class Matcher:
    def __init__(self, product_database_dir: str,
                 feature_matcher = matcher.loftr_matcher,
                 device=Device.cpu):
        if feature_matcher == matcher.loftr_matcher:
            print("Using loftr")
            classifier = Loftr_Classifier(product_database_dir=product_database_dir,
                                          product_database_path=f"{product_database_dir}/product_database.csv",
                                          device=device)
        else:
            print("Using sift")
            classifier = Sift_Classifier()
        self.classifier = classifier

class Pipeline:
    def __init__(self, model_path='/data/ia_tech_conaprole/repos/Dense-Object-Detection/weights/best.pt',
                    output_dir="./results",
                    device=Device.cuda,
                    matcher=matcher.loftr_matcher,
                    debug=True,
                    num_processes = 1,
                    product_database_dir= "/data/ia_tech_conaprole/cluster/matcher_classifier_four_products"):

        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.device = device
        if device == Device.cuda:
            self.yolo_device = select_device('0' if torch.cuda.is_available() else 'cpu')
        else:
            self.yolo_device = select_device('cpu')

        self.classifier_parameters = dict(product_database_dir= product_database_dir,
                                          feature_matcher=matcher,
                                          device=device)

        self.model = self.load_yolov5_model(model_path)

        self.num_processes = num_processes
        self.size = 640
        self.conf_thres = 0.6
        self.iou_thres = 0.45
        self.score_th = 60

    def load_yolov5_model(self, model_path):
        model = attempt_load(model_path, device=self.yolo_device)
        return model

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def draw_bounding_boxes(self, image, boxes, color = (255, 0, 0), thickness = 3):
        for box in boxes:
            x1, y1, x2, y2 = box
            #convert coordenates to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        return image

    def write_image(self, image):
        output_path = f"{self.output_dir}/detection_{self.output_prefix}_nms_{self.conf_thres}_iou_{self.iou_thres}_size_{self.size}.png"
        image_r = cv2.resize(image, (640, 640))
        image_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_r)

    def yolov5_inference(self, img):

        Hi,Wi, _ = img.shape
        img_r = resize_image_using_pil_lib(img, self.size, self.size)
        Hf, Wf, _ = img_r.shape
        #img = np.array(image_r)
        img_r = img_r[:, :, ::-1].transpose(2, 0, 1)
        img_r = np.ascontiguousarray(img_r)
        img_r = torch.from_numpy(img_r).to(self.yolo_device)
        img_r = img_r.float() / 255.0
        img_r = img_r.unsqueeze(0)

        # Run the YOLOv5 model on the image
        pred = self.model(img_r)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # convert to numpy
        pred = [x.detach().cpu().numpy() for x in pred]
        # convert to int
        pred = [x.astype(int) for x in pred]
        # Post-process the output and draw bounding boxes on the image
        boxes = []
        confidences = []
        class_ids = []
        for det in pred:
            if det is not None and len(det):
                # Scale the bounding box coordinates to the original image size
                det[:, [0,2]] = det[:, [0,2]] * Wi / Wf
                det[:, [1,3]] = det[:, [1,3]] * Hi / Hf
                for *xyxy, conf, cls in det:
                    boxes.append(xyxy)
                    confidences.append(conf.item())
                    class_ids.append(int(cls.item()))

        image = np.array(img)

        if self.debug:
            image = self.draw_bounding_boxes(image.copy(), boxes)
            self.write_image(image)

        return boxes

    @staticmethod
    def classify_box(classifier, image, box):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_image = image[y1:y2, x1:x2]
        product_name, _, score = classifier.predict(bbox_image)
        if product_name is None:
            return None

        return product_name

    def classifier_multiprocessing(self, image, boxes):
        #from multiprocessing import Pool, freeze_support
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')

        classifier = Matcher(**self.classifier_parameters).classifier
        product_name_list = []
        #freeze_support()

        with mp.Pool(processes=self.num_processes) as pool:
             product_name_list = pool.starmap(self.classify_box, [(classifier, image, box) for box in boxes])
        #     #product_name_list = [product_name for product_name in product_name_list if product_name is not None]


        for product_name, bbox in zip(product_name_list, boxes):
            if product_name is None:
                continue
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if self.debug:
                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{self.output_prefix}_{x1}_{y1}_{x2}_{y2}.png"
                bbox_image = image[y1:y2, x1:x2]
                bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bbox_image)

    def classifier(self, image, boxes):
        classifier = Matcher(**self.classifier_parameters).classifier
        product_name_list = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_image = image[y1:y2, x1:x2]
            prefix = f"{self.output_prefix}_{x1}_{y1}_{x2}_{y2}"
            product_name, _ , score = classifier.predict(bbox_image, prefix)
            if product_name is None:
                continue

            if score < self.score_th:
                continue

            product_name_list.append(product_name)

            if self.debug:
                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{prefix}_{int(score*100)}.png"
                print(f"Product Name: {product_name} Ratio Inference: ", score)
                bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bbox_image)

        return product_name_list



    def compute_metrics(self, res):
        #count frequency of each product
        from collections import Counter
        import pandas as pd

        counter = Counter(res)
        df = pd.DataFrame(counter.items(), columns=["Product", "Frequency"])
        #save df as html
        self.output_metrics_path = f"{self.output_dir}/{self.output_prefix}_metrics.html"
        df.to_html(self.output_metrics_path)
        self.df_metrics = df
        return self.output_metrics_path

    def print_metrics(self):
        print(self.df_metrics)


    def main(self, image_path):
        self.output_dir = Path(self.output_dir) / Path(image_path).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = Path(image_path).stem

        image = self.preprocess_image(image_path)
        boxes = self.yolov5_inference(image)
        res = self.classifier(image, boxes)
        res = self.compute_metrics(res)
        self.print_metrics()
        return res


def main(image_path):
    pipeline = Pipeline()
    pipeline.main(image_path)

    return

if __name__ == "__main__":
    image_path = "./assets/WhatsApp Image 2024-05-24 at 12.00.13 (2).jpeg"
    image_path = "./assets/IMG_9149.png"
    #image_path = "./assets/IMG_9156.png"
    image_path = "images_for_demo/matcher/WhatsApp Image 2024-05-24 at 15.42.24 (2).jpeg"
    image_path = "images_for_demo/matcher/WhatsApp Image 2024-05-24 at 12.17.32 (10).jpeg"
    main(image_path)