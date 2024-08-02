import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from classifier import Loftr_Classifier, Sift_Classifier, Device
from feature_matching import AisleProductMatcher
from product_identification_app import ProductIdentificationApp

class matcher:
    loftr_matcher = 0
    sift_matcher = 1
class WeakLabelling:
    """
    Class for weak labelling of bounding boxes on images. Annotations in labelme format are in the directory
    annotations_dir. Images are in the directory images_dir. The output is saved in the directory output_dir.
    To each bounding box within each image a category is assigned. The category is determined by Loftr_Classifier.
    """
    def __init__(self, annotations_dir: str, images_dir: str, output_dir: str,
                 product_database_dir: str,
                 feature_matcher = matcher.loftr_matcher,
                 device=Device.cpu):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        if feature_matcher == matcher.loftr_matcher:
            print("Using loftr")
            classifier = Loftr_Classifier(product_database_dir=product_database_dir,
                                          product_database_path=f"{product_database_dir}/product_database.csv",
                                          device=device)
        else:
            print("Using sift")
            classifier = Sift_Classifier()
        self.classifier = classifier




    def weak_labelling(self):
        """
        Assigns a category to each bounding box within each image. The category is determined by Loftr_Classifier.
        """
        images_path = (list(self.images_dir.rglob("*.png")) + list(self.images_dir.rglob("*.jpg"))
                       + list(self.images_dir.rglob("*.jpeg")))
        for image_path in images_path:
            image = cv2.imread(str(image_path))
            # convert to annotation name
            annotation_name = str(image_path.stem).replace(" ", "_") + "_modified.json"
            annotations_path = self.annotations_dir / annotation_name
            if not annotations_path.exists():
                continue
            all_bbox_annotations = AisleProductMatcher.load_labelme_rectangle_shapes(annotations_path)
            for idx, bbox_annotation in enumerate(all_bbox_annotations):
                y_min, y_max, x_min, x_max = ProductIdentificationApp._get_rectangle_top_bottom(bbox_annotation)
                bbox_image = image[y_min:y_max, x_min:x_max]
                prefix = f"{image_path.stem}_{idx}"
                files = self.output_dir.rglob(f"{prefix}*")
                if len(list(files)) > 0:
                    print(prefix)
                    continue
                product_name, _ , score = self.classifier.predict(bbox_image, prefix)
                if product_name is None:
                    continue

                output_dir = self.output_dir / product_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{prefix}_{int(score)}.png"
                cv2.imwrite(str(output_path), bbox_image)

        return

def main(device, matcher, root="/data/ia_tech_conaprole/cluster", product_database_dir=""):
    annotations_dir = root / Path("annotations")
    images_dir = root / Path("images")
    output_dir = root / Path("output/weak_labelling")
    output_dir.mkdir(parents=True, exist_ok=True)
    weak_labelling = WeakLabelling(annotations_dir, images_dir, output_dir, product_database_dir,
                                   device = device, feature_matcher = matcher)
    weak_labelling.weak_labelling()
    return

if __name__ == "__main__":
    import argparse
    # add device as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default = Device.cpu)
    #add matcher as argument. By defult loftr
    parser.add_argument("--matcher", type=int, default = matcher.loftr_matcher)
    #add root dataset
    parser.add_argument("--root", type=str, default = "/data/ia_tech_conaprole")
    parser.add_argument("--product_database_dir", type=str, default = "")
    args = parser.parse_args()

    main(args.device, args.matcher, args.root, args.product_database_dir)


