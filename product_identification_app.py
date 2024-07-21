import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from feature_matching import AisleProductMatcher
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

    def __init__(self, image_path, bbox_annotation_path):
        self.product_id = None
        self.product_path = None
        self.image_path = image_path
        self.all_bbox_annotations = AisleProductMatcher.load_labelme_rectangle_shapes(bbox_annotation_path)
        self.product_bbox_annotations = []
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

        return int(y_min), int(y_max), int(x_min), int(x_max)
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
        self.figp = None
        self.fig = None


        from matplotlib.widgets import Button
        self.mode = ProductIdentificationApp.select_bbox
        figsize = (15,15)
        if self.fig is not None:
            plt.close(self.fig)

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

        #add a button for select the product image and store the path in self.product_path
        self.axbutton5 = plt.axes([0.4, 0.025, 0.1, 0.04])
        self.button5 = Button(self.axbutton5, 'Select Product')
        self.button5.on_clicked(self.select_product)





    def select_product(self, event):
        #open a file dialog to select the product image


        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        root_dir = "/data/ia_tech_conaprole/dataset"
        self.product_path = askopenfilename(initialdir=root_dir, title="Select Product Image", filetypes=[("Image files", "*.png *.jpg *.webp")])
        #select product id
        product_metadata_path = "./output/product_frequency.csv"
        product_metadata = pd.read_csv(product_metadata_path)
        #get row that match self.product_path
        #product_metadata = product_metadata[product_metadata["product_path"] == self.product_path]
        #if product_metadata.shape[0] == 0:
        #    print("Product not found")
        #    return

        #self.product_id = product_metadata["product_id"].values[0]
        #create an unique integer id for the product based on the product_path.stem
        self.product_id = Path(self.product_path).stem
        print(f"Product id: {self.product_id}")

        self.show_product()
        self.product_bbox_annotations = []
        #self.__build_ui()
        image_to_display = self._draw_bbox_annotations()
        self.ax.imshow(image_to_display)
        plt.draw()



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
        if self.product_path is None:
            return

        image = cv.imread(self.product_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.figp is not None:
            plt.close(self.figp)
        self.figp, self.axp = plt.subplots()
        self.axp.imshow(image)
        plt.show()

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

        if Path(self.output_file).exists():
            #df.to_csv(self.output_file, mode='a', header=False, index=False)
            df.to_csv(self.output_file, mode='a', header=False, index=False)

        else:
            df.to_csv(self.output_file, index=False)
        print("save annotations")

        self.mode = ProductIdentificationApp.annotation_made
        self.draw_title()
        plt.draw()



    def run(self, processed_product_annotations_path):
        self.output_file = processed_product_annotations_path
        plt.show()
        pass


def main(image_path, bbox_annotation_dir, output_dir):
    #image_path = Path("./assets/WhatsApp Image 2024-05-27 at 09.23.32 (1)/WhatsApp Image 2024-05-27 at 09.23.32 (1).jpeg")
    image_path = Path(image_path)
    annotation_name = str(image_path.stem).replace(" ", "_") + "_modified.json"
    bbox_annotation_path = f"{bbox_annotation_dir}/{annotation_name}"

    app = ProductIdentificationApp(image_path, bbox_annotation_path)
    #output_path = "./assets/WhatsApp Image 2024-05-27 at 09.23.32 (1)/annotations.csv"
    Path(f"{output_dir}/{image_path.stem}").mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{image_path.stem}/annotations.csv"
    app.run(output_path)

    return

if __name__ == "__main__":
    #add as arguments the image_path and the bbox_annotation_dir and the output_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./assets/WhatsApp Image 2024-05-27 at 09.23.32 (1)/WhatsApp Image 2024-05-27 at 09.23.32 (1).jpeg")
    parser.add_argument("--bbox_annotation_dir", type=str, default="/data/ia_tech_conaprole/dataset/modified")
    parser.add_argument("--output_dir", type=str, default="./assets/")
    args = parser.parse_args()
    main(args.image_path, args.bbox_annotation_dir, args.output_dir)