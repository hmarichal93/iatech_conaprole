import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import pandas as pd

def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data

def compute_descriptors(image):
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)

    #surf = cv.xfeatures2d.SURF_create(400)
    #kp, des = surf.detectAndCompute(image, None)

    return kp, des

class Product:
    def __init__(self, image_path, idx):
        self.image_path = image_path
        self.image = cv.imread(str(image_path))#, cv.IMREAD_GRAYSCALE)
        self.kp, self.des = compute_descriptors(self.image)
        self.id = idx

    def __eq__(self, other):
        return self.id == other.id

class AisleProduct(Product):
    def __init__(self, image_path, idx, bbox):
        super().__init__(image_path, idx)
        self.bbox = bbox


    def draw_bbox(self, image, channel=0):
        (y1, x1), (y2, x2) = self.bbox
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image.shape[0], y2)
        x2 = min(image.shape[1], x2)
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #color rectangle in blue over image
        image[int(y1):int(y2), int(x1):int(x2), channel] = 255
        return image

def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object, keep_ratio= True) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    #flag = Image.ANTIALIAS
    flag = Image.Resampling.LANCZOS
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    return im_r


class AisleProductMatcher:
    def __init__(self, aisle_image_path, aisle_annotations_path, product_store, output_dir= None):

        self.product_store = product_store
        self.aisle_annotations_name = Path(aisle_annotations_path).stem
        self.aisle_image_name = Path(aisle_image_path).stem

        self.output_dir = Path(output_dir) / self.aisle_image_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aisle_image = cv.imread(aisle_image_path)
        self.aisle_annotations = self.load_labelme_rectangle_shapes(aisle_annotations_path)
        self.aisle_products = self.get_id_for_each_product_in_the_aisle()


    def get_id_for_each_product_in_the_aisle(self, debug=False):
        aisle_products = []
        self.aisle_products_dir = self.output_dir / "aisle"
        if debug:
            self.aisle_products_dir.mkdir(parents=True, exist_ok=True)

        for idx, bbox in enumerate(self.aisle_annotations):
            bbox_image = self.get_bbox_image(bbox)
            if debug:
                aisle_debug_image = self.aisle_image.copy()
                H, W, _ = aisle_debug_image.shape

            try:
                product, score = self.match_product(bbox_image)
                if product:
                    aisle_products.append(AisleProduct(product.image_path, product.id, bbox))
                    if debug:
                        print(f"Matched product {product.id} with score {score}")
                        aisle_products[-1].draw_bbox(aisle_debug_image, channel=2)
                        product_image = product.image

                        factor = 5
                        heigth, width = H // factor, W // factor
                        product_image = resize_image_using_pil_lib(product_image, heigth, width)
                        hp,wp,_ = product_image.shape
                        #aisle_debug_image[0:hp, 0:wp] = 0
                        aisle_debug_image[0:hp, 0:wp] =  product_image
                        #write in green the score
                        cv2.putText(aisle_debug_image, f"{score:.2f}", (int(bbox[0][1]), int(bbox[0][0] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 5)

                        #save debug image
                        output_path = self.aisle_products_dir / f"bbox_{idx}.png"
                        cv.imwrite(str(output_path), aisle_debug_image)

            except cv2.error:
                continue


        return aisle_products

    def load_labelme_rectangle_shapes(self, labelme_json_path):
        try:
            json_content = load_json(labelme_json_path)
            l_rings = []
            for ring in json_content['shapes']:
                if ring['shape_type'] == "rectangle":
                    l_rings.append(np.array(ring['points'])[:, [1, 0]].tolist())

        except FileNotFoundError:
            l_rings = []

        return l_rings

    def get_bbox_image(self, bbox, size=0.5):
        (y1, x1), (y2, x2) = bbox
        height_bbox = int(y2) - int(y1)
        width_bbox = int(x2) - int(x1)
        y_min = max(0, int(y1) - size*height_bbox)
        y_max = min(self.aisle_image.shape[0], int(y2) + size*height_bbox)
        x_min = max(0, int(x1) - size*width_bbox)
        x_max = min(self.aisle_image.shape[1], int(x2) + size*width_bbox)
        return self.aisle_image[int(y_min):int(y_max), int(x_min):int(x_max)]
        #x1, y1, x2, y2 = bbox
        #return self.aisle_image[int(y1):int(y2), int(x1):int(x2)]

    def save_annotations_in_csv(self):
        csv_path = self.output_dir / f"{self.aisle_image_name}.csv"
        rows = []
        for aisle_product in self.aisle_products:
            H,W,_ = self.aisle_image.shape
            product_id = aisle_product.id
            bbox = aisle_product.bbox
            (y1, x1), (y2, x2) = bbox
            y1 = int(max(0, y1))
            x1 = int(max(0, x1))
            y2 = int(min(H, y2))
            x2 = int(min(W, x2))
            rows.append([x1, y1, x2, y2, product_id, W, H])

        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2", "product_id", "W", "H"])
        df.to_csv(csv_path, index=False)


    def draw_products_in_aisle(self, debug=False):
        product_store_ids = [product.id for product in self.aisle_products]
        #get unique ids
        H, W, _ = self.aisle_image.shape
        product_store_ids = list(set(product_store_ids))
        for idx  in product_store_ids:
            output_path = self.output_dir / f"product_{idx}.png"
            debug_image = self.aisle_image.copy()
            for aisle_product in self.aisle_products:
                if aisle_product.id == idx:
                    if debug:
                        debug_image = aisle_product.draw_bbox(debug_image)





            if debug:
                #draw product in the aisle image
                product = self.product_store.get_product_by_id(idx)
                img = product.image
                #reshape image to H/10 and W/10
                factor = 5
                heigth, width = H//factor, W//factor
                img  = resize_image_using_pil_lib(img, heigth, width)
                #resize keeping aspect ratio

                hp,wp,_ = img.shape
                #debug_image[0:hp, 0:wp] = 0
                debug_image[0:hp, 0:wp] =  img
                #write image
                cv.imwrite(str(output_path), debug_image)

    def match_product(self, img, metcher_th = 10):
        # BFMatcher with default params
        bf = cv.BFMatcher()
        kp, des = compute_descriptors(img)
        best_match = None
        best_match_score = 0
        for product in self.product_store.products:
            matches = bf.knnMatch(des, product.des, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matcher_percentage = len(good) #/ len(product.kp)
            if len(good) > best_match_score and matcher_percentage> metcher_th:
                best_match = product
                best_match_score = len(good)
        return best_match, best_match_score



def get_image_files(directory):
    return [str(file) for file in Path(directory).rglob('*') if file.suffix in {'.png', '.webp', '.jpg'}]



class ProductStore:
    def __init__(self, products_dir="/data/ia_tech_conaprole/dataset/Yogures - Conaprole"):
        self.product_dir = products_dir
        self.products = self.load_products()

    def load_products(self):
        products_path = get_image_files(self.product_dir)
        product_list = []
        for idx, product_path in enumerate(products_path):
            product = Product(product_path, idx)
            product_list.append(product)

        return product_list

    def get_product_by_id(self, idx):
        for product in self.products:
            if product.id == idx:
                return product
        return None
def using_sift_for_classifier_object(annotation_labels_dir="/data/ia_tech_conaprole/dataset/modified/",
                                     aisle_images_dir="/data/ia_tech_conaprole/dataset/fotos de pdv",
                                     output_dir="./output"):
    products_cona = ProductStore()
    from tqdm import tqdm
    #get all json files
    json_files = Path(annotation_labels_dir).rglob("*.json")
    json_files_list = list(json_files)
    for annotations_aisle_path in tqdm(json_files_list, total=len(json_files_list)):
        json_content = load_json(annotations_aisle_path)
        image_ailse_path = json_content["imagePath"]
        image_name = "WhatsApp" + Path(image_ailse_path).stem.split("WhatsApp")[1]
        #print(image_name)
        image_ailse_path = Path(aisle_images_dir).rglob(f"{image_name}.jpeg")
        image_ailse_path = image_ailse_path.__next__()

        aisle_product_matcher = AisleProductMatcher(image_ailse_path, annotations_aisle_path, products_cona,
                                                    output_dir=output_dir)
        aisle_product_matcher.save_annotations_in_csv()

    return

def count_product_frequency_in_aisle(output_dir="./output"):
    products_cona = ProductStore()
    csv_files = Path(output_dir).rglob("*.csv")
    count_dict = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        product_ids = df["product_id"].values
        product_ids, counts = np.unique(product_ids, return_counts=True)
        print(f"File {csv_file.name}")
        for product_id, count in zip(product_ids, counts):
            if product_id in count_dict:
                count_dict[product_id] += count
            else:
                count_dict[product_id] = count

    for product_id in count_dict.keys():
        product = products_cona.get_product_by_id(product_id)
        count_dict[product_id] = (product.image_path, count_dict[product_id])

    #create dataframe
    items = [item[1] for item in count_dict.items()]
    index = [item[0] for item in count_dict.items()]
    df = pd.DataFrame(items, columns=["product_path", "count"],index=index)
    #sort df by count descending

    df = df.sort_values(by="count", ascending=False)
    df.to_csv(f"{output_dir}/product_frequency.csv", index=False)


    return


if __name__=="__main__":
    #using_sift_for_classifier_object()
    count_product_frequency_in_aisle()