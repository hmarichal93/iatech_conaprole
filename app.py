import cv2
import yolov5

from pathlib import Path

from weak_labelling import matcher
from classifier import Device, Loftr_Classifier, Sift_Classifier

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
                    num_processes = 1 ):
        self.model = self.load_yolov5_model(model_path)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.device = device

        self.classifier_parameters = dict(product_database_dir="/data/ia_tech_conaprole/cluster/matcher_classifier_2",
                                          feature_matcher=matcher,
                                          device=device)

        self.num_processes = num_processes

    def load_yolov5_model(self, model_path):
        model = yolov5.load(model_path)
        # set model parameters
        model.conf = 0.4  # NMS confidence threshold
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
        output_path = f"{self.output_dir}/detection_{self.output_prefix}_nms_{self.model.conf}_iou_{self.model.iou}_size_{self.model.size}.png"
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
        if self.device == Device.cuda:
            return self.classifier_multiprocessing(image, boxes)

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

            product_name_list.append(product_name)

            if self.debug:
                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{prefix}_{int(score)}.png"
                bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bbox_image)

        return product_name_list



    def compute_metrics(self):

        pass

    def print_metrics(self, res):
        pass

    def main(self, image_path):
        self.output_dir = Path(self.output_dir) / Path(image_path).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = Path(image_path).stem

        image = self.preprocess_image(image_path)
        boxes = self.yolov5_inference(image)
        res = self.classifier(image, boxes)
        res = self.compute_metrics(res)
        self.print_metrics(res)
        return res


def main(image_path):

    pipeline = Pipeline()
    pipeline.main(image_path)

    return

if __name__ == "__main__":
    image_path = "./assets/WhatsApp Image 2024-05-24 at 12.00.13 (2).jpeg"
    #image_path = "./assets/IMG_9149.png"
    #image_path = "./assets/IMG_9156.png"
    main(image_path)