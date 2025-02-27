# IA Tech - Conaprole Challenge: Share of Space

## Problem Statement
The objective of this challenge is to estimate the share of space of a given product in a set of images. The share of space is defined as the percentage of the total area occupied by a product in a given image. The product is identified by its brand name. The images are taken from different points of sale (POS) and the products are placed on shelves. The images are taken from different angles and distances, and the products can be partially occluded by other products or objects. The images are provided by CONAPROLE and the products are from the CONAPROLE brand.
## Images
![Image 1](assets/detection__image_1722826729.8273664_full_image.png)
![Image 2](assets/detection__image_1722826729.8273664_full_image_conaprole.png)

## Table

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product</th>
      <th>Frequency</th>
      <th>Share of Space</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Others</td>
      <td>164</td>
      <td>47.13</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Conaprole</td>
      <td>184</td>
      <td>52.87</td>
    </tr>
  </tbody>
</table>

## Installation

### Dense-Object-Detection submodule
Pretrained YoloV5 model over [SKU110k](https://paperswithcode.com/dataset/sku110k) dataset. The model is trained on 110k images and 1.7M annotations.
Download the pretrained model from [here](https://drive.google.com/file/d/1BRlXZD9MqYAYYnciMRQ50Mht9kBncf0l/view).
```bash
git clone git@github.com:suryanshgupta9933/Dense-Object-Detection.git
mv Dense-Object-Detection dod
```

### Package installation
1. Create a conda environment and install the required packages.
```bash
conda create -n conaprole python=3.7
conda activate conaprole
pip install -r requierments.txt
```
Install DenseObjectDetection requirements as well
```bash
cd dob
pip install -r requirements.txt
```
In case of error with cuda11 and nvidia-cublas, uninstall nvidia-cublas-cu11
```bash
pip uninstall nvidia_cublas_cu11
```

## Usage
```bash
./run.sh
```

### Dataset
Put your image set in the dataset folder. You can use a symbolic link to the actual location of the dataset.
Use the following command to create a symbolic link:

```bash
ln -s /path/to/your/dataset/* dataset
```

Folder structure should be as follows:
* images - images provided by CONAPROLE
* annotations - human labels for the images.


## Generate gt annotations

1.0 download yolov5 baseline predicctions: https://github.com/hmarichal93/iatech_conaprole/releases/tag/v1.0
2.0 run the following command
```bash
python main --images_dir DATASETPATH/fotos de pdv --output_dir PREDICCIONES_YOLO --mannually_modify_predictions
```


### Annotate product within an image
```bash
python product_annotation.py --image_path DATASETPATH/fotos de pdv/WhatsApp Image 2024-05-27 at 09.23.32 (1).jpeg --output_dir DATASETPATH/ --bbox_annotation_dir DATASETPATH/

```

## Google Cloud commands

### Connect to gcloud
```bash
gcloud auth application-default login
```

### Copy files to gcloud
```bash
gcloud compute scp --project conaprole --zone us-west1-b --recurse  /data/ia_tech_conaprole/repos/Dense-Object-Detection/weights/best.pt  deeplearning-1-vm:~/iatech_conaprole
```
