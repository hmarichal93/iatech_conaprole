import cv2
import cv2 as cv
import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
from feature_matching import ProductStore
from pathlib import Path
from feature_matching import AisleProductMatcher, resize_image_using_pil_lib, there_is_intersection
from product_identification_app import ProductIdentificationApp
import pandas as df
from tqdm import tqdm

import torch
import time

class Product:
    def __init__(self, name, images_list):
        self.name = name
        self.image_list = images_list

    def get_dataframe(self):
        #generate dataframe with header name and image path
        df = pd.DataFrame(columns=["name", "image_path"])
        for image_path in self.image_list:
            #df = df.append({"name": self.name, "image_path": image_path}, ignore_index=True)
            df.loc[df.shape[0]] = [self.name, str(image_path)]

        return df




class Classifier(ABC):
    def __init__(self, product_database_path=None, product_database_dir=None):
        if not Path(product_database_path).exists():
            self.__build_product_database(product_database_dir, product_database_path)

        self.df_products = pd.read_csv(product_database_path)



    def __build_product_database(self, product_database_dir, product_database_path):
        """
        Directory with images of each product. In each product folder, there are several images from the same product.
        :param product_database_dir:
        :return:
        """
        product_database_dir = Path(product_database_dir)
        products = []
        file_extension = [".jpeg", ".webp", ".png", ".jpg"]
        for product_dir in product_database_dir.iterdir():
            if not product_dir.is_dir():
                continue

            product_name = product_dir.name
            images = list(product_dir.rglob(f"*.*"))


            product = Product(product_name, images)
            products.append(product)

        #save the database
        df = pd.DataFrame(columns=["name", "image_path"])
        for product in products:
            #df = df.append(product.get_dataframe(), ignore_index=True)
            #using df.loc for appending df
            data = {'name': product.get_dataframe().name.values.tolist(),
                    'image_path': product.get_dataframe().image_path.values.tolist()}
            df_product = pd.DataFrame(data)
            #concatenate dataframes df and df_product
            if df_product.shape[0] == 0:
                print(f"Product {product.name} has no images")
                continue
            df = pd.concat([df, df_product], axis=0)
            #for idx, row in product.get_dataframe().iterrows():
            #    df.loc[df.shape[0]] = [row['name'], row.image_path]

        df.to_csv(product_database_path, index=False)
        return

    @abstractmethod
    def compute_similarity(self, image1, image2):
        """
        Compute the similarity between two images.
        This method must be implemented by all subclasses of Classifier.
        """
        pass

    def predict(self, image, image_name=None):
        best_similarity = 0
        best_product = None
        best_image_template = None
        for idx, row in tqdm(self.df_products.iterrows(),desc=f"Predicting-{image_name}", total=self.df_products.shape[0]):
            image_template_path = row.image_path

            image_template = cv.imread(image_template_path)
            #compute the similarity
            similarity = self.compute_similarity(image, image_template)
            if similarity > best_similarity:
                best_similarity = similarity
                best_product = row['name']
                best_image_template = image_template_path

        return best_product, best_image_template, best_similarity

from kornia.feature import LoFTR
import kornia.feature as KF
import kornia as K
from kornia_moons.viz import draw_LAF_matches

class Device:
    cuda = 0
    cpu = 1
class Loftr_Classifier(Classifier):
    def __init__(self, product_database_path=None, product_database_dir=None, device = Device.cuda ):
        super().__init__(product_database_path, product_database_dir)
        if device == Device.cuda:
            print("Using cuda")
            self.matcher =  LoFTR(pretrained="indoor_new").cuda()
        else:
            print("Using cpu")
            self.matcher =  LoFTR(pretrained="indoor_new")

        self.device = device

    def compute_similarity(self, image1, image2):
        """Compute matching score between two images using LoFTR."""
        if image1 is None or image2 is None:
            return 0
        #convert numpy array to tensor
        image_1_path = "/tmp/image1.png"
        cv2.imwrite(image_1_path, image1)
        image_2_path = "/tmp/image2.png"
        cv2.imwrite(image_2_path, image2)


        if self.device == Device.cuda:
            image1 = K.io.load_image(image_1_path, K.io.ImageLoadType.RGB32)[None, ...]
            image2 = K.io.load_image(image_2_path, K.io.ImageLoadType.RGB32)[None, ...]
            img1 = K.geometry.resize(image1, (480, 640), antialias=True)
            img2 = K.geometry.resize(image2, (480, 640), antialias=True)
            img1 = img1.cuda()
            img2 = img2.cuda()
        else:
            image1 = K.io.load_image(image_1_path, K.io.ImageLoadType.RGB32)[None, ...].cpu().numpy()
            image2 = K.io.load_image(image_2_path, K.io.ImageLoadType.RGB32)[None, ...].cpu().numpy()
            img1 = K.geometry.resize(image1, (480, 640), antialias=True).cpu().numpy()
            img2 = K.geometry.resize(image2, (480, 640), antialias=True).cpu().numpy()

        # compute the matches
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img2),
        }
        #matches = self.matcher(input_dict)
        #to = time.time()
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)
        #tf = time.time()
        #print(f"Time: {tf - to}")
        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        if mkpts0.shape[0] < 4:
            return 0
        try:
            #to = time.time()
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            inliers = inliers > 0
            #tf = time.time()
            #print(f"Time: {tf - to}")
        except cv2.error as e:
            inliers = []
            pass


        return len(inliers)


class Sift_Classifier(Classifier):
    def __init__(self, product_database_path=None, product_database_dir=None):
        super().__init__(product_database_path, product_database_dir)
        self.matcher =  LoFTR(pretrained="indoor_new").cuda()

    def compute_descriptors(self, image):
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)

        # surf = cv.xfeatures2d.SURF_create(400)
        # kp, des = surf.detectAndCompute(image, None)

        return kp, des
    def compute_similarity(self, image1, image2):
        """Compute matching score between two images using LoFTR."""
        kp1, des1 = self.compute_descriptors(image1)
        kp2, des2 = self.compute_descriptors(image2)
        bf = cv.BFMatcher()
        best_match = None
        best_match_score = 0
        matches = bf.knnMatch(des1, des2, k=2)

        #convert numpy array to tensor
        image_1_path = "/tmp/image1.png"
        cv2.imwrite(image_1_path, image1)
        image_2_path = "/tmp/image2.png"
        cv2.imwrite(image_2_path, image2)

        image1 = K.io.load_image(image_1_path, K.io.ImageLoadType.RGB32)[None, ...]
        image2 = K.io.load_image(image_2_path, K.io.ImageLoadType.RGB32)[None, ...]

        img1 = K.geometry.resize(image1, (480, 640), antialias=True).cuda()
        img2 = K.geometry.resize(image2, (480, 640), antialias=True).cuda()

        # compute the matches
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img2),
        }
        #matches = self.matcher(input_dict)

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        if mkpts0.shape[0] < 4:
            return 0
        try:
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            inliers = inliers > 0
        except cv2.error as e:
            inliers = []
            pass


        return len(inliers)









class Sift_Classifier:
    def __init__(self, product_store):
        self.product_store = product_store


    def compute_descriptors(self, image):
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)

        # surf = cv.xfeatures2d.SURF_create(400)
        # kp, des = surf.detectAndCompute(image, None)

        return kp, des

    def brute_force_match(self, des1, des2):
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        return good

    def magsac_find_homography(self, kp1, kp2, good):
        #use magsac to find homography
        #
        #transform kp1 and kp2 to numpy arrays
        #get from good the points that are in kp1 and kp2
        kp1 = np.array([kp1[m[0].queryIdx].pt for m in good])
        kp2 = np.array([kp2[m[0].trainIdx].pt for m in good])
        try:
            H, mask = cv2.findHomography(kp1.T, kp2.T, cv2.USAC_MAGSAC)
        except cv2.error as e:
            return np.array([])
        #get the inliers
        inliers = np.where(mask == 1)[0]
        return inliers

    def match_images_by_sift_features(self, img1, img2):
        kp1, des1 = self.compute_descriptors(img1)
        kp2, des2 = self.compute_descriptors(img2)

        good = self.brute_force_match(des1, des2)
        stimator = len(good)
        #inliers = self.magsac_find_homography(kp1, kp2, good)
        #stimator = inliers.shape[0] / len(good)
        return stimator

    def run_inferece(self, image):
        max_matches = 0
        predicted_product = None
        for idx, product in enumerate(self.product_store.products):
            matches = self.match_images_by_sift_features(image, product.image)
            if matches > max_matches:
                max_matches = matches
                predicted_product = product

        #print (f"Matches: {max_matches} ")
        return predicted_product

    def get_product_by_id(self, idx):
        for product in self.product_store.products:
            if product.id == idx:
                return product
        return None


def test_classifier():


    product_store = ProductStore()
    classifier = Sift_Classifier(product_store)
    image_path = "assets/WhatsApp Image 2024-05-27 at 09.23.32 (1)/WhatsApp Image 2024-05-27 at 09.23.32 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-27 at 12.50.36 (1)/WhatsApp Image 2024-05-27 at 12.50.36 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-29 at 09.01.25 (1)/WhatsApp Image 2024-05-29 at 09.01.25 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-29 at 09.01.16 (3)/WhatsApp Image 2024-05-29 at 09.01.16 (3).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-27 at 11.44.47/WhatsApp Image 2024-05-27 at 11.44.47.jpeg"
    image_path = "./assets/WhatsApp Image 2024-05-27 at 12.50.35 (1)/WhatsApp Image 2024-05-27 at 12.50.35 (1).jpeg"
    #image_path = "./assets/IMG_9140/IMG_9140.png"
    output_dir = Path(image_path).parent / "output"
    gt_prod_id_file = Path(image_path).parent / "annotations.csv"
    gt_prod_id_elements = df.read_csv(gt_prod_id_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv.imread(image_path)
    H, W, _ = image.shape
    p_f = 10
    #extract bounding boxes
    image_path = Path(image_path)
    bbox_annotation_dir = "/data/ia_tech_conaprole/dataset/modified"
    #bbox_annotation_dir = "./assets/IMG_9140/"
    annotation_name = str(image_path.stem).replace(" ", "_") + "_modified.json"
    bbox_annotation_path = f"{bbox_annotation_dir}/{annotation_name}"
    bbox_list = AisleProductMatcher.load_labelme_rectangle_shapes(bbox_annotation_path)
    debug_image = image.copy()
    import time
    to = time.time()
    for idx, bbox in tqdm(enumerate(bbox_list), total=len(bbox_list)):
        #label file
        gt_prod_id = [row.product_id for idx, row in gt_prod_id_elements.iterrows() if there_is_intersection(bbox, [[row.y1, row.x1], [row.y2, row.x2]])]
        if len(gt_prod_id) == 0:
            #print(f"bbox idx {idx} Cannot find ground truth product id")
            continue

        y_min, y_max, x_min, x_max = ProductIdentificationApp._get_rectangle_top_bottom(bbox)
        y_min = max(0, y_min)
        y_max = min(H-1, y_max)
        x_min = max(0, x_min)
        x_max = min(W-1, x_max)
        bbox_image = image[y_min:y_max, x_min:x_max]
        product = classifier.run_inferece(bbox_image)

        h_p = H // p_f
        w_p = W // p_f
        product_global_image = np.zeros_like(image)
        if product:
            product_image = resize_image_using_pil_lib(product.image, h_p, w_p)
            h_p, w_p, _ = product_image.shape
            product_global_image[:h_p, :w_p] = product_image

        bbox_global_image = np.zeros_like(image)
        bbox_global_image[y_min:y_max, x_min:x_max] = bbox_image

        #concatenate images
        aux_image = np.concatenate([image, bbox_global_image, product_global_image], axis=1)
        output_path = output_dir / f"output_{idx}.jpg"
        cv.imwrite(output_path, resize_image_using_pil_lib(aux_image, 800, 800))


        gt_prod_id = gt_prod_id[0]
        if product and product.id == gt_prod_id:
            print(f"Product {product.id} was correctly identified")
            debug_image[y_min:y_max, x_min:x_max,  1] = 125
        else:
            print(f"Product {product.id} was not correctly identified as {gt_prod_id}")
            debug_image[y_min:y_max, x_min:x_max, 2] = 125

    cv.imwrite(output_dir / "debug.jpg", debug_image)
    print(f"Time: {time.time() - to}")
    return

def evaluate_classifier(image_target_dir="./assets/IMG_9140"):
    image_target_dir = "./assets/WhatsApp Image 2024-05-27 at 12.50.35 (1)/"
    image_target_dir = "assets/WhatsApp Image 2024-05-27 at 11.44.47/"
    image_target_dir = Path(image_target_dir)
    output_dir = image_target_dir / "output"
    gt_prod_id_file = image_target_dir / "annotations.csv"
    gt_prod_id_elements = df.read_csv(gt_prod_id_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = next(image_target_dir.glob("*.jpeg"))
    image = cv.imread(image_path
                      )
    H, W, _ = image.shape
    p_f = 10
    # extract bounding boxes
    bbox_annotation_dir = image_target_dir
    bbox_annotation_dir = Path("/data/ia_tech_conaprole/dataset/modified")

    annotation_name = str(image_path.stem).replace(" ", "_") + "_modified.json"
    bbox_annotation_path = f"{bbox_annotation_dir}/{annotation_name}"
    bbox_list = AisleProductMatcher.load_labelme_rectangle_shapes(bbox_annotation_path)
    debug_image = image.copy()
    #instantiate classifier
    root = "/data/ia_tech_conaprole/dataset/matcher_classifier"
    classifier = Loftr_Classifier(product_database_dir=root,
                                  product_database_path=f"{root}/product_database.csv",
                                  device=Device.cpu)

    for idx, bbox in tqdm(enumerate(bbox_list), total=len(bbox_list)):
        # label file
        gt_prod_id = [row.product_id for idx, row in gt_prod_id_elements.iterrows() if
                      there_is_intersection(bbox, [[row.y1, row.x1], [row.y2, row.x2]])]
        if len(gt_prod_id) == 0:
            # print(f"bbox idx {idx} Cannot find ground truth product id")
            continue
        ##########################

        #########################
        y_min, y_max, x_min, x_max = ProductIdentificationApp._get_rectangle_top_bottom(bbox)
        y_min = max(0, y_min)
        y_max = min(H - 1, y_max)
        x_min = max(0, x_min)
        x_max = min(W - 1, x_max)
        bbox_image = image[y_min:y_max, x_min:x_max]
        product_name, image_template = classifier.predict(bbox_image)
        product_image  = cv.imread(image_template)
        h_p = H // p_f
        w_p = W // p_f
        product_global_image = np.zeros_like(image)
        if product_name:
            product_image = resize_image_using_pil_lib(product_image, h_p, w_p)
            h_p, w_p, _ = product_image.shape
            product_global_image[:h_p, :w_p] = product_image

        bbox_global_image = np.zeros_like(image)
        bbox_global_image[y_min:y_max, x_min:x_max] = bbox_image

        # concatenate images
        aux_image = np.concatenate([image, bbox_global_image, product_global_image], axis=1)
        output_path = output_dir / f"output_{idx}.jpg"
        cv.imwrite(output_path, resize_image_using_pil_lib(aux_image, 800, 800))

        gt_prod_id = gt_prod_id[0]
        if product_name and product_name == gt_prod_id:
            print(f"Product {product_name} was correctly identified")
            debug_image[y_min:y_max, x_min:x_max, 1] = 125
        else:
            print(f"Product {product_name} was not correctly identified as {gt_prod_id}")
            debug_image[y_min:y_max, x_min:x_max, 2] = 125

    cv.imwrite(output_dir / "debug.jpg", debug_image)

    return






if __name__ == "__main__":
    #test_classifier()
    evaluate_classifier()



