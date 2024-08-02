import cv2
import pandas as pd

from pathlib import Path

def main():
    root = "./assets/IMG_9140"
    image_path = Path(root) / "IMG_9140.png"
    annotations_path = Path(root) / "annotations.csv"
    image = cv2.imread(str(image_path))
    df = pd.read_csv(annotations_path)

    for idx, row in df.iterrows():
        x1, y1, x2, y2 = row.x1, row.y1, row.x2, row.y2
        cv2.rectangle(image, (y1, x1), (y2, x2), (255, 0, 0), 1)
        #coloring
        image[y1:y2, x1:x2, 1] = 125

    cv2.imwrite("./assets/output.png", image)
    return

if __name__ == "__main__":
    main()
