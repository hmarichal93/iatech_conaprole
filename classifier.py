import cv2
import cv2 as cv
import numpy as np


class Classifier:
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

        print (f"Matches: {max_matches} ")
        return predicted_product

    def get_product_by_id(self, idx):
        for product in self.product_store.products:
            if product.id == idx:
                return product
        return None


def test_classifier():
    from feature_matching import ProductStore
    from pathlib import Path
    from feature_matching import AisleProductMatcher, resize_image_using_pil_lib, there_is_intersection
    from product_identification_app import ProductIdentificationApp
    import pandas as df
    from tqdm import tqdm

    product_store = ProductStore()
    classifier = Classifier(product_store)
    image_path = "assets/WhatsApp Image 2024-05-27 at 09.23.32 (1)/WhatsApp Image 2024-05-27 at 09.23.32 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-27 at 12.50.36 (1)/WhatsApp Image 2024-05-27 at 12.50.36 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-29 at 09.01.25 (1)/WhatsApp Image 2024-05-29 at 09.01.25 (1).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-29 at 09.01.16 (3)/WhatsApp Image 2024-05-29 at 09.01.16 (3).jpeg"
    image_path = "assets/WhatsApp Image 2024-05-27 at 11.44.47/WhatsApp Image 2024-05-27 at 11.44.47.jpeg"
    output_dir = Path(image_path).parent / "output"
    gt_prod_id_file = Path(image_path).parent / "annotations.csv"
    gt_prod_id_elements = df.read_csv(gt_prod_id_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv.imread(image_path
                      )
    H, W, _ = image.shape
    p_f = 10
    #extract bounding boxes
    image_path = Path(image_path)
    bbox_annotation_dir = "/data/ia_tech_conaprole/dataset/modified"
    annotation_name = str(image_path.stem).replace(" ", "_") + "_modified.json"
    bbox_annotation_path = f"{bbox_annotation_dir}/{annotation_name}"
    bbox_list = AisleProductMatcher.load_labelme_rectangle_shapes(bbox_annotation_path)
    debug_image = image.copy()
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
        cv.imwrite(output_path, aux_image)


        gt_prod_id = gt_prod_id[0]
        if product and product.id == gt_prod_id:
            #print(f"Product {product.id} was correctly identified")
            debug_image[y_min:y_max, x_min:x_max,  1] = 125
        else:
            #print(f"Product {product.id} was not correctly identified")
            debug_image[y_min:y_max, x_min:x_max, 2] = 255

    cv.imwrite(output_dir / "debug.jpg", debug_image)



if __name__ == "__main__":
    test_classifier()



