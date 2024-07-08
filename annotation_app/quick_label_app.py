"""
App to quick label the images stored in dataset/images. The app allow to label the images using the labelme tool
(https://github.com/labelmeai/labelme) . Additionally, allows to skip images that are not good enought.
This information is stored in the file dataset/labels.csv. This file has the
following columns: image_path, label_path, skip. The label column is the label of the image and the skip column is a boolean that
indicates if the image should be skipped or not. The app will show the images that have not been labeled yet and that are not
skipped. The app will show the image and the labelme tool to label the image. The app will update the file dataset/labels.csv with the respective
image path, label path and skip value. The app will also store the label file in the dataset/labels folder.
"""

import os
import cv2
import pandas as pd
from pathlib import Path
import customtkinter

class LabelTool(customtkinter.CTk):
    def __init__(self, dataset_dir="dataset/images/fotos de pvd/", labels_dir="dataset/labels",
                 labels_file="dataset/labels/labels.csv"):

        super().__init__()

        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.labels_file = labels_file
        self.labels = pd.read_csv(self.labels_file)
        self.labels = self.labels.fillna('')
        self.labels = self.labels.set_index('image_path')
        self.labels = self.labels.sort_index()
        self._images_paths = self.__get_images_paths()

        self.__build_ui()



    def __build_ui(self):
        self.title("Label Tool")
        self.geometry("800x600")
        self._img_label = customtkinter.CTkLabel(self)
        self._img_label.pack()

        # Create a frame to hold the buttons and position it at the top of the window
        self._button_frame = customtkinter.CTkFrame(self)
        self._button_frame.pack(side="top")  # Adjusted to position the frame at the top

        # Add the buttons to the frame instead of directly to the window
        self._labelme_button = customtkinter.CTkButton(self._button_frame, text="Label", command=self.label)
        self._skip_button = customtkinter.CTkButton(self._button_frame, text="Skip", command=self.skip)

        # Pack the buttons inside the frame, side by side
        self._labelme_button.pack(side="left", padx=10)
        self._skip_button.pack(side="right", padx=10)

        self._current_image_index = 0




    def __get_images_paths(self):
        images_paths = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.jpg'):
                    images_paths.append(os.path.join(root, file))
        return images_paths

    def __get_next_image(self):
        pass

    def label(self):
        pass

    def skip(self):
        pass


if __name__ == "__main__":
    app = LabelTool()
    app.mainloop()


