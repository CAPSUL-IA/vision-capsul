import requests
import shutil
import json
import os

url = "https://www.kaggle.com/api/v1/datasets/download/kagkimch/crowdhuman-coco"

response = requests.get(url, stream=True)

if response.status_code == 200:
    with open("crowdhuman.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Archivo ZIP descargado correctamente.")
else:
    print(f"Error {response.status_code}: {response.text}")

shutil.unpack_archive(
    "crowdhuman.zip",
    "data/"
)

annotations_path = "data/CrowdHuman/annotation_"
image_path = "data/CrowdHuman/Images"
size_path = "data/CrowdHuman/id_hw_"

output_labels = "data/labels"
os.makedirs(output_labels, exist_ok=True)

output_images = "data/images"
os.makedirs(output_images, exist_ok=True)

for task in ["train", "val"]:
    task_path = f"{annotations_path}{task}.odgt"

    output_labels_task = f"{output_labels}/{task}"
    os.makedirs(output_labels_task, exist_ok=True)

    output_images_task = f"{output_images}/{task}"
    os.makedirs(output_images_task, exist_ok=True)

    with open(f"{size_path}{task}.json", 'r', encoding='utf-8') as size_file:
        data_size = json.loads(size_file.read())
        with open(task_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())

                image_id = data["ID"]
                gtboxes = data["gtboxes"]

                if f"{image_id}.jpg" in os.listdir(image_path):
                    shutil.move(os.path.join(image_path,f"{image_id}.jpg"),os.path.join(output_images_task,f"{image_id}.jpg"))

                height, width = data_size[image_id]
                with open(f"{output_labels_task}/{image_id}.txt", "w") as txt_file:
                    for box in gtboxes:
                        tag = box.get("tag", "")
                        extra = box.get("extra", {})
                        ignore = extra.get("ignore", 0)

                        if tag == "person" and ignore == 0:
                            vbox = box.get("vbox", None)
                            if vbox:
                                x, y, w_box, h_box = vbox

                                x_c = (x + w_box / 2) / width
                                y_c = (y + h_box / 2) / height
                                w_n = w_box / width
                                h_n = h_box / height

                                if (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w_n <= 1 and 0 < h_n <= 1):
                                    txt_file.write(f"0 {x_c} {y_c} {w_n} {h_n}\n")

if os.path.exists("crowdhuman.zip"):
    os.remove("crowdhuman.zip")

label2id =  {
        0: "person"
    }

with open('data/label2id.json', 'w') as f:
    json.dump(label2id, f, indent=4)

