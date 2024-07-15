import os
from pathlib import Path
import json

def mannually_modify_predictions(dataset_dir, prediction_dirs, annotated_files="./labeled_files.csv"):
    #get all json files
    import pandas as pd

    if Path(annotated_files).exists():
        pd_file = pd.read_csv(annotated_files)
    else:
        pd_file = pd.DataFrame(columns=["json_file", "modified"])

    json_files = Path(prediction_dirs).rglob("*.json")
    output_dir = Path(prediction_dirs).parent / "modified"
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_path in json_files:
        get_location_where_substring_appears = [json_path.name for annotated_path in list(pd_file["json_file"].values) if json_path.name in annotated_path]
        is_json_path_already_annotated = len(get_location_where_substring_appears) > 0
        if is_json_path_already_annotated:
            continue

        #######fix the json file to point to the correct image
        json_content = json.load(open(json_path))
        image_name = Path(json_content['imagePath']).name
        image_path = Path(dataset_dir).rglob(f"{image_name}").__next__()
        json_content['imagePath'] = str(image_path)
        with open(json_path, 'w') as json_file:
            json.dump(json_content, json_file)
        #############################################################################################
        json_new_name = output_dir / (json_path.stem.replace(" ","_") + "_modified.json")
        command = f"cp \"{str(json_path)}\" \"{str(json_new_name)}\""
        print(command)
        os.system(command)
        #command = (f"cd {str(output_dir)} && " +
        command = f"labelme \"{str(json_path)}\" --output \"{str(json_new_name)}\""
        print(command)
        os.system(command)

        #pd_file = pd_file.append({"json_file": json_path.name, "modified": json_new_name.name}, ignore_index=True)
        #add to the dataframe but not append
        pd_file.loc[pd_file.shape[0]] = [str(json_path), str(json_new_name)]

        pd_file.to_csv(annotated_files, index=False)

    return



if __name__ == "__main__":
    #add argument for making predictions over the full dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="/data/ia_tech_conaprole/dataset/fotos de pdv")
    parser.add_argument("--output_dir", type=str, default="/home/henry/Documents/repo/fing/others/challenges/iatech_conaprole/dataset/labels/yolov5_inference_results")
    parser.add_argument("--resize", type=int, default=640)
    #add flag make_predictions
    parser.add_argument("--make_predictions", action="store_true")
    #add flag mannually_modify_predictions
    parser.add_argument("--mannually_modify_predictions", action="store_true")

    args = parser.parse_args()

    if args.make_predictions:
        from inference import make_prediction_over_full_dataset
        make_prediction_over_full_dataset(args.images_dir, args.output_dir, args.resize)

    if args.mannually_modify_predictions:
        mannually_modify_predictions(args.images_dir, args.output_dir)

