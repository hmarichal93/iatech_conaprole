import cv2
#import yolov5

from pathlib import Path
import numpy as np
import torch
from weak_labelling import matcher
from classifier import Device, Loftr_Classifier, Sift_Classifier
from feature_matching import resize_image_using_pil_lib
from PIL import Image
from  tqdm import tqdm

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
                    product_database_dir= "/data/ia_tech_conaprole/cluster/matcher_classifier_four_products",
                    conf_th=0.6, iou_th=0.45,
                    split_image=False):

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
        self.conf_thres = conf_th
        self.iou_thres = iou_th
        self.score_th = 60
        self.split_image = split_image

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

    def write_image(self, image, image_prefix=None):
        if image_prefix is None:
            output_path = f"{self.output_dir}/detection_{self.output_prefix}.png"
        else:
            output_path = f"{self.output_dir}/detection__{self.output_prefix}_{image_prefix}.png"

        image_r = image#cv2.resize(image, (640, 640))
        image_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR)
        print(output_path)
        cv2.imwrite(output_path, image_r)

    def yolov5_inference(self, img, image_prefix=None):

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

        #if self.debug:
        image = self.draw_bounding_boxes(image.copy(), boxes)
        self.write_image(image, image_prefix=image_prefix)

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

    def classifier(self, image, boxes, classifier_type="automl"):
        from classifier import AutoML_Classifier
        classifier = Matcher(**self.classifier_parameters).classifier if classifier_type != "automl" else AutoML_Classifier(default_dir=self.output_dir)

        product_name_list = []
        for box in tqdm(boxes, desc="Classifying boxes"):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_image = image[y1:y2, x1:x2]
            prefix = f"{self.output_prefix}_{x1}_{y1}_{x2}_{y2}"
            product_name, _ , score = classifier.predict([bbox_image], [prefix])
            # if product_name is None:
            #     continue
            #
            # if score < self.score_th and classifier_type != "automl":
            #     continue

            if (classifier_type == "automl" and score < self.conf_thres and score < self.score_th and
                    classifier_type != "automl" and  product_name is None):
                product_name_list.append(product_name)
                continue

            product_name_list.append(product_name)

            if self.debug:
                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{prefix}_{int(score*100)}.png"
                #print(f"Product Name: {product_name} Ratio Inference: ", score)
                bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bbox_image)

        return product_name_list

    def classifier_batch(self, image, boxes, classifier_type="automl"):
        from classifier import AutoML_Classifier
        classifier = AutoML_Classifier(default_dir=self.output_dir)

        product_name_list = []
        bbox_images = []
        prefixes = []
        for box in tqdm(boxes, desc="Classifying boxes"):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox_image = image[y1:y2, x1:x2]
            prefix = f"{self.output_prefix}_{x1}_{y1}_{x2}_{y2}"
            bbox_images.append(bbox_image)
            prefixes.append(prefix)
        product_names, _, scores = classifier.batch_prediction(bbox_images, prefixes, self.conf_thres)
        #product_names, _, scores = classifier.predict(bbox_images, prefixes, self.conf_thres)



        if self.debug:
            for idx, product_name in enumerate(product_names):
                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                score = scores[idx]
                prefix = prefixes[idx]
                bbox_image = bbox_images[idx]
                output_path = output_dir / f"{prefix}_{int(score*100)}.png"
                #print(f"Product Name: {product_name} Ratio Inference: ", score)
                bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bbox_image)

        return product_name_list



    def compute_metrics(self, res):
        #count frequency of each product
        from collections import Counter
        import pandas as pd
        res = [product_name for product_name in res if product_name is not None]
        res_p = []
        for p in res:
            if p is None:
                res_p.append("None")
            else:
                res_p.append(p)
        res = res_p

        counter = Counter(res)
        df = pd.DataFrame(counter.items(), columns=["Product", "Frequency"])
        #compuse share of space of each product. product_frequency / total_frequency
        df["Share of Space"] = df["Frequency"] / df["Frequency"].sum()
        df["Share of Space"] = df["Share of Space"] * 100
        #share of space column only 2 decimal places
        df["Share of Space"] = df["Share of Space"].apply(lambda x: round(x, 2))
        self.output_metrics_path = f"{self.output_dir}/{self.output_prefix}_metrics.html"

        df.to_html(self.output_metrics_path)

        df_s = pd.DataFrame(columns=["Product", "Frequency", "Share of Space"])

        #add row with product with Conaprole prefix frequency
        conaprole_frequency = df[df["Product"].str.contains("Conaprole")]["Frequency"].sum()
        conaprole_share = conaprole_frequency / df["Frequency"].sum()
        #append using .loc method
        df_s.loc[len(df_s)] = ["Conaprole", conaprole_frequency, conaprole_share]
        #compute row with product withou Conaprole prefix frequency
        others_frequency = df[~df["Product"].str.contains("Conaprole")]["Frequency"].sum()
        others_share = others_frequency / df["Frequency"].sum()
        df_s.loc[len(df_s)] = ["Others", others_frequency, others_share]
        #multiply by 100 to get percentage
        df_s["Share of Space"] = df_s["Share of Space"] * 100
        #limit to 2 decimal places

        #invert the order of the dataframe by index
        df_s = df_s.iloc[::-1]
        #set index column name
        #df.index.name = "Index"
        #save df as html
        self.output_metrics_path = f"{self.output_dir}/{self.output_prefix}_global.html"
        #share of space column only 2 decimal places
        df_s["Share of Space"] = df_s["Share of Space"].apply(lambda x: round(x, 2))
        df_s.to_html(self.output_metrics_path)
        self.df_metrics = df_s
        return self.output_metrics_path

    def print_metrics(self, products_name, boxes, image):
        print(self.df_metrics)
        image_debug = image.copy()
        for product_name, box in zip(products_name, boxes):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if product_name is not None and "Conaprole" in product_name:
                image_debug = cv2.rectangle(image_debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
                image_debug[y1:y2, x1:x2,1] = 150
            else:
                image_debug = cv2.rectangle(image_debug, (x1, y1), (x2, y2), (255, 0, 0), 3)

        image_debug = cv2.putText(image_debug, f"Conaprole Share of Space: {self.df_metrics.loc[self.df_metrics['Product'] == 'Conaprole']['Share of Space'].values[0]}%",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        self.write_image(image_debug, image_prefix="full_image_conaprole")


    @staticmethod
    def there_is_overlap(box1, box2):
        from shapely.geometry import Polygon
        x1, y1, x2, y2 = box1
        box1 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        x3, y3, x4, y4 = box2
        box2 = Polygon([(x3, y3), (x4, y3), (x4, y4), (x3, y4)])
        if box1.intersects(box2):
            return True
        return False

    @staticmethod
    def rectangles_intersect(rect1, rect2):
        """
        Check if two rectangles intersect.

        Each rectangle is defined by a tuple (x1, y1, x2, y2) where:
        (x1, y1) is the top-left corner
        (x2, y2) is the bottom-right corner
        """
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2

        # Check if one rectangle is to the left of the other
        if x1_max < x2_min or x2_max < x1_min:
            return False

        # Check if one rectangle is above the other
        if y1_max < y2_min or y2_max < y1_min:
            return False

        return True

    @staticmethod
    def compute_union_boxes(box, box_new):
        "compute the union between both  boxes using shapely polygon"
        from shapely.geometry import Polygon
        x1, y1, x2, y2 = box
        #get corner points of the box
        box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        box1_p = Polygon(box)
        x3, y3, x4, y4 = box_new
        box_new = [(x3, y3), (x4, y3), (x4, y4), (x3, y4)]

        box2_p = Polygon(box_new)
        if box1_p.area> box2_p.area:
            return x1, y1, x2, y2
        else:
            return x3, y3, x4, y4






    def split_image_and_run_inference(self, image):
        """
        Split image in patches and run inference on each patch. Then merge the results
        :param image:
        :return:
        """
        #TODO fix bug in the intersection of the patches
        #1.0 split image in patches with a factor of 0.25

        Hi, Wi, _ = image.shape
        num_patches = 2

        patch_size_x = int(Wi / num_patches)
        num_patches_x = num_patches# int(Wi / patch_size_x)
        patch_size_y = int(Hi  / num_patches)
        num_patches_y = num_patches #int(Hi / patch_size_y )
        #overlapping
        overlapping = 0.05
        #split the image in patches
        boxes_orig = []
        image_debug = image.copy()
        for i in range(num_patches_x):
            x1 = int(i * patch_size_x * (1 - overlapping))
            x2 = x1 + patch_size_x
            x2 = min(x2, Wi - 1) #if i < num_patches_x - 1 else Wi -2
            for j in range(num_patches_y):
                y1 = int(j * patch_size_y * (1 - overlapping))
                y2 = y1 + patch_size_y
                y2 = min(y2, Hi - 1) #if j < num_patches_y - 1 else Hi -2
                #print(f"Processing patch {i} {j}. {x1=} {x2=} {y1=} {y2=}")

                #y1 = min(y1,Hi)
                patch = image[y1:y2, x1:x2]
                #draw in image_debug rectangle of the patch
                image_debug = cv2.rectangle(image_debug, (x1, y1), (x2, y2), (255, 0, 0), 3)
                boxes = self.yolov5_inference(patch, image_prefix = f"{i}_{j}")
                #revert the coordinate of the bboxes to the original
                boxes = [[x1 + box[0], y1 + box[1], x1 + box[2], y1 + box[3]] for box in boxes]

                if len(boxes_orig) == 0:
                    boxes_orig.extend(boxes)
                    continue
                print("Checking overlapping")
                boxes_orig_copy = boxes_orig.copy()
                boxes_to_add = []
                print(len(boxes_orig))
                for box_new in boxes:
                    for box in boxes_orig:
                        if not self.rectangles_intersect(box, box_new):
                            if box_new not in boxes_to_add:
                                boxes_to_add.append(box_new)
                        else:
                            box_extension = self.compute_union_boxes(box, box_new)
                            if box in boxes_orig_copy:
                                boxes_orig_copy.remove(box)
                            if box_extension not in boxes_to_add:
                                boxes_to_add.append(box_extension)

                boxes_orig = boxes_orig_copy
                boxes_orig.extend(boxes_to_add)

        print('Finish inference')
        image = self.draw_bounding_boxes(image.copy(), boxes_orig)
        self.write_image(image, image_prefix="full_image")
        self.write_image(image_debug, image_prefix="image_patches")

        return boxes_orig




    def main(self, image_path):
        text = "split" if self.split_image else "full"
        self.output_dir = Path(self.output_dir) / f"{Path(image_path).stem}_{text}"
        if Path(self.output_dir).exists():
            import os
            os.system(f"rm -r {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = Path(image_path).stem

        image = self.preprocess_image(image_path)
        if self.split_image:
            #split the image in patches
            boxes = self.split_image_and_run_inference(image)
        else:
            boxes = self.yolov5_inference(image)
            image_debug = self.draw_bounding_boxes(image.copy(), boxes)
            self.write_image(image_debug, image_prefix="full_image")
        products_name = self.classifier(image, boxes)
        #res = self.classifier_batch(image, boxes)
        res = self.compute_metrics(products_name)
        self.print_metrics(products_name, boxes, image)
        return self.output_dir


def main(image_path,conf_th=0.6, iou_th=0.4, debug=True, model_path=None, split_image=False):
    pipeline = Pipeline(conf_th=conf_th, iou_th=iou_th,debug=debug, model_path=model_path, split_image=split_image)
    output_dir  = pipeline.main(image_path)

    return output_dir

if __name__ == "__main__":
    import argparse
    #image_path and model_path are the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="images_for_demo/autml/IMG_9140.png")
    parser.add_argument("--model_path", type=str, default="./weights/best.pt")
    parser.add_argument("--conf_th", type=float, default=0.6)
    parser.add_argument("--iou_th", type=float, default=0.4)
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--split_image", type=bool, default=False)

    args = parser.parse_args()
    main(args.image_path, args.conf_th, args.iou_th, args.debug, args.model_path, split_image=args.split_image)
