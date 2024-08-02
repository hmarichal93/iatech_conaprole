import numpy as np
import cv2
from pathlib import Path


def split_image_in_patches(image, patch_size=640):
    """
    Split an image in patches of size patch_size with overlapping
    :param image: image to split
    :param patch_size: size of the patch
    :return: list of patches
    """
    patches = []
    image_height, image_width = image.shape[:2]
    for y in range(0, image_height, patch_size):
        for x in range(0, image_width, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


def main(image_dir="/data/ia_tech_conaprole/fotos_joselo", output_dir = "/data/ia_tech_conaprole/fotos_joselo_splitted"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    images_path = Path(image_dir).rglob("*.png")
    for image_path in images_path:
        image = cv2.imread(image_path)
        patches = split_image_in_patches(image, patch_size = 2000)
        for idx, patch in enumerate(patches):
            output_patch_path = Path(output_dir) / f"{image_path.stem}_{idx}.png"
            cv2.imwrite(output_patch_path, patch)

    return

if __name__=="__main__":
    main()