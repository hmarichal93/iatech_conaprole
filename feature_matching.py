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
        self.id = Path(image_path).stem

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

    @staticmethod
    def load_labelme_rectangle_shapes(labelme_json_path):
        try:
            json_content = load_json(labelme_json_path)
            l_rings = []
            for ring in json_content['shapes']:
                if ring['shape_type'] == "rectangle":
                    l_rings.append(np.array(ring['points'], dtype=int)[:, [1, 0]].tolist())

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
    statics_path = f"{output_dir}/product_frequency.csv"
    for csv_file in csv_files:
        if str(csv_file).endswith("product_frequency.csv"):
            continue
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
    # add a new column product id
    df["product_id"] = df.index

    df = df.sort_values(by="count", ascending=False)
    df.to_csv(statics_path, index=False)


    return




def check_first_iteration_product_id(first_iteration_product_frequency_path="./output/product_frequency.csv", output_dir = "./output_product_id"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.csv"
    if status_path.exists():
        df_status = pd.read_csv(status_path)
    else:
        df_status = pd.DataFrame(columns=["product_id", "image_path"])

    df = pd.read_csv(first_iteration_product_frequency_path)
    product_ids = df["product_id"].values
    product_ids = list(product_ids)
    bbox_annotation_dir = Path("/data/ia_tech_conaprole/dataset/modified") #bbox annotations processed by labelme but without product identifiaation
    product_annotations_dir = Path("./output/") #product annotations generated by sift
    product_annotations_paths = list(product_annotations_dir.rglob("*.csv"))
    product_annotations_paths = [path for path in product_annotations_paths if not str(path).endswith("product_frequency.csv")]

    images_dir = Path("/data/ia_tech_conaprole/dataset/fotos de pdv")
    product_path = df["product_path"].values
    product_path = list(product_path)
    for p_path, p_id in zip(product_path, product_ids):
        #1. Get all the images with occurence of the product
        images_path = []
        for product_annotations_path in product_annotations_paths:
            df = pd.read_csv(product_annotations_path)
            product_ids = df["product_id"].values
            if p_id in product_ids:
                aisle_image_name = product_annotations_path.parent.name
                image_ailse_path = Path(images_dir).rglob(f"{aisle_image_name}.jpeg")
                image_ailse_path = image_ailse_path.__next__()
                images_path.append(str(image_ailse_path))
        from tqdm import tqdm
        #2. iterate over the images
        for img_path in tqdm(images_path, total = len(images_path)):
            #check if the produc_id and the image_path are in the status file
            if df_status[(df_status["product_id"] == p_id) & (df_status["image_path"] == img_path)].shape[0] > 0:
                continue
            #2.1 get all bbox annotations
            image_name = Path(img_path).stem
            annotation_name = f"{image_name}_modified.json".replace(" ","_")
            annotation_path = bbox_annotation_dir / annotation_name
            all_bbox_annotations = AisleProductMatcher.load_labelme_rectangle_shapes(annotation_path)
            #all_bbox_annotations = [convert_top_up_botton_down_bbox_to_yolov5(bbox) for bbox in all_bbox_annotations]
            #2.2 get all the bbox annotations initially identified as the product
            product_ids_path = product_annotations_dir / image_name / f"{image_name}.csv"
            df = pd.read_csv(product_ids_path)
            #filter by product id
            product_bbox_annotations = df[df["product_id"] == p_id]
            product_bbox_annotations = [[[row.y1,row.x1],[row.y2,row.x2]] for idx, row in product_bbox_annotations.iterrows() ]

            # 2.3 remove from all_bbox_annotations the bbox that are in product_bbox_annotations
            for p_bbox in product_bbox_annotations:
                all_bbox_annotations = [bbox for bbox in all_bbox_annotations if not there_is_intersection(bbox, p_bbox)]

            #2.4 run the app
            app = ProductIdentificationApp(p_path, p_id,  img_path, all_bbox_annotations, product_bbox_annotations)
            output_processed_dir = output_dir / image_name
            output_processed_dir.mkdir(parents=True, exist_ok=True)
            processed_product_annotations_path = output_processed_dir / f"{image_name}_{p_id}.csv"
            app.run(processed_product_annotations_path)

            #2.5 save the status

            row = {"product_id": p_id, "image_path": img_path}
            #add row dic to dataframe df_status with a different method than append
            df_status.loc[df_status.shape[0]] = row
            df_status.to_csv(status_path, index=False)






    return product_ids


class ProductIdentificationApp:
    """
    App for product identification. The app will show the image and the bbox annotations. The user will have to select the
    bbox for the product that were not identified by the sift algorithm (product_bbox_annotations) from the original
    bounding boxes (all_bbox_annotations). The product that must be identified is defined by the product_path.
    Additionaly, wrong identifications can be removed. App is developed using matplotlib with 3 modes (buttons) and mouse interaction
    - bottun 1: select bbox for missing product
    - bottun 2: remove bbox from wrong identification
    - bottun 3: save the annotations
    """
    select_bbox = 1
    remove_bbox = 2
    annotation_made = 3

    def __init__(self, product_path, product_id ,  image_path, all_bbox_annotations, product_bbox_annotations):
        self.product_id = product_id
        self.product_path = product_path
        self.image_path = image_path
        self.all_bbox_annotations = all_bbox_annotations
        self.product_bbox_annotations = product_bbox_annotations
        self.image = cv.imread(str(image_path))
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.selected_bbox = None
        self.processed_product_annotations_path = None
        self.__build_ui()

    @staticmethod
    def _get_rectangle_top_bottom(bbox):
        (y1, x1), (y2, x2) = bbox
        y_max = max(y1, y2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        x_min = min(x1, x2)
        return y_min, y_max, x_min, x_max
    def _draw_bbox_annotations(self):
        image_to_display = self.image.copy()
        for bbox in self.all_bbox_annotations:
            y_min, y_max, x_min, x_max = self._get_rectangle_top_bottom(bbox)
            self.__draw_bbox_rectangle(image_to_display, x_max, x_min, y_max, y_min, color = (0, 255, 0))

        for bbox in self.product_bbox_annotations:
            y_min, y_max, x_min, x_max = self._get_rectangle_top_bottom(bbox)
            self.__draw_bbox_rectangle(image_to_display, x_max, x_min, y_max, y_min)
            #color rectangle in blue over image
            image_to_display[int(y_min):int(y_max), int(x_min):int(x_max), 2] = 255

        return image_to_display
    @staticmethod
    def __draw_bbox_rectangle(image, x_max, x_min, y_max, y_min, color = (255, 0, 0)):
        cv.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
    def __build_ui(self):
        #import Button from matplotlib
        import matplotlib
        #matplotlib.use('WebAgg')
        matplotlib.use('TkAgg')
        #display matplotlib windows in browser



        from matplotlib.widgets import Button
        self.mode = ProductIdentificationApp.select_bbox
        figsize = (15,15)
        self.fig, self.ax = plt.subplots(figsize=figsize)

        image_to_display = self._draw_bbox_annotations()
        self.ax.imshow(image_to_display)
        self.draw_title()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        #add buttons
        self.axbutton1 = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.axbutton2 = plt.axes([0.7, 0.025, 0.1, 0.04])
        self.axbutton3 = plt.axes([0.6, 0.025, 0.1, 0.04])
        #add a four button to show in a new window the product image (product_path)
        self.axbutton4 = plt.axes([0.5, 0.025, 0.1, 0.04])
        self.button1 = Button(self.axbutton1, 'Select bbox')
        self.button2 = Button(self.axbutton2, 'Remove bbox')
        self.button3 = Button(self.axbutton3, 'Save annotations')
        self.button4 = Button(self.axbutton4, 'Close App')
        self.button1.on_clicked(self.select_bbox)
        self.button2.on_clicked(self.remove_bbox)
        self.button3.on_clicked(self.save_annotations)
        self.button4.on_clicked(self.close_app)

        self.show_product()

    def close_app(self, event):
        if self.mode == ProductIdentificationApp.annotation_made:
            plt.close(self.fig)
            plt.close(self.figp)
            plt.close("all")
            print("close app")
        else:
            print("You must save the annotations before close the app")

    def show_product(self, event=None):
        #display in a new figure self.product_path image
        image = cv.imread(self.product_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        self.figp, self.axp = plt.subplots()
        self.axp.imshow(image)
        #plt.show()

    def onclick(self, event):
        print("click")
        if event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata
        if self.mode == ProductIdentificationApp.select_bbox:
            for idx, bbox in enumerate(self.all_bbox_annotations):
                if self._click_within_bbox(bbox, x, y):
                    self.product_bbox_annotations.append(bbox)
                    self.all_bbox_annotations.pop(idx)
                    break

            #remove bbox from all_bbox_annotations



        elif self.mode == ProductIdentificationApp.remove_bbox:
            #to witch bbox the click belongs
            for idx, bbox in enumerate(self.product_bbox_annotations):
                if self._click_within_bbox(bbox, x, y):
                    bbox = self.product_bbox_annotations.pop(idx)
                    self.all_bbox_annotations.append(bbox)
                    break

        self.draw_title()
        self.ax.imshow(self._draw_bbox_annotations())
        plt.draw()


    def _click_within_bbox(self, bbox, x, y):
        y_min, y_max, x_min, x_max = self._get_rectangle_top_bottom(bbox)
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        else:
            return False
    def draw_title(self):
        mode = "Select" if self.mode == ProductIdentificationApp.select_bbox else "Remove"

        if self.mode == ProductIdentificationApp.annotation_made:
            mode = "Annotation made"

        self.ax.set_title(f"Select bbox for the product: Mode: {mode}")

    def select_bbox(self, event):
        print("select bbox")
        self.mode = ProductIdentificationApp.select_bbox
        self.draw_title()
        plt.draw()


    def remove_bbox(self, event):
        print("remove bbox")
        self.mode = ProductIdentificationApp.remove_bbox
        self.draw_title()
        plt.draw()

    def save_annotations(self, event):

        #save the annotations
        #header
        # x1,y1,x2,y2,product_id,W,H
        rows = []
        H, W, _ = self.image.shape
        for bbox in self.product_bbox_annotations:
            (y1, x1), (y2, x2) = bbox
            rows.append([x1, y1, x2, y2, self.product_id, W, H])

        print(self.output_file)
        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2", "product_id", "W", "H"])
        #replace columns negative values with 0
        df["x1"] = df["x1"].apply(lambda x: max(0, x))
        df["y1"] = df["y1"].apply(lambda x: max(0, x))
        df["x2"] = df["x2"].apply(lambda x: max(0, x))
        df["y2"] = df["y2"].apply(lambda x: max(0, x))
        #replace if x2 > W with W -1
        df["x2"] = df["x2"].apply(lambda x: min(W -1, x))
        df["x1"] = df["x1"].apply(lambda x: min(W - 1, x))
        #replace if y2 > H with H -1
        df["y2"] = df["y2"].apply(lambda x: min(H -1, x))
        df["y1"] = df["y1"].apply(lambda x: min(H - 1, x))

        df.to_csv(self.output_file, index=False)
        print("save annotations")

        self.mode = ProductIdentificationApp.annotation_made
        self.draw_title()
        plt.draw()



    def run(self, processed_product_annotations_path):
        self.output_file = processed_product_annotations_path
        plt.show()
        pass




def there_is_intersection(bbox1, bbox2, iou_th=0.7):
    (y1, x1), (y2, x2) = bbox1
    (y3, x3), (y4, x4) = bbox2
    #check if there is intersection
    from shapely.geometry import Polygon
    polygon1 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    polygon2 = Polygon([(x3, y3), (x4, y3), (x4, y4), (x3, y4)])
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersection / union
    if iou > iou_th:
        return True
    else:
        return False




def convert_top_up_botton_down_bbox_to_yolov5(bbox):
    (y1, x1), (y2, x2) = bbox
    height = y2 - y1
    width = x2 - x1
    x_center = x1 + width/2
    y_center = y1 + height/2
    return (int(x_center), int(y_center), int(width), int(height))

def merge_all_labels_in_one_file(output_dir="./output_product_id"):
    csv_files = Path(output_dir).rglob("*.csv")
    csv_files = [path for path in csv_files if not str(path).endswith("status.csv")]
    data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        image_name = Path(csv_file).stem.split("_")[0]
        for idx, row in df.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            product_id = row["product_id"]
            W = row["W"]
            H = row["H"]
            data.append([image_name, x1, y1, x2, y2, product_id, W, H])

    df = pd.DataFrame(data, columns=["image_name", "x1", "y1", "x2", "y2", "product_id", "W", "H"])
    df.to_csv(f"{output_dir}/all_labels.csv", index=False)

    #compute product frequency
    df["product_id"].hist()
    plt.show()

    return



if __name__=="__main__":
    #using_sift_for_classifier_object()
    #count_product_frequency_in_aisle()
    #Product identification app
    #check_first_iteration_product_id()
    #count frequency
    merge_all_labels_in_one_file()