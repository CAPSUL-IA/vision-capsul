import requests
import shutil
import pandas as pd
import os
from PIL import Image

url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"

# Hacer la solicitud y guardar el archivo
response = requests.get(url, stream=True, verify=False)

if response.status_code == 200:
    with open("eurosat.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Archivo ZIP descargado correctamente.")
else:
    print(f"Error {response.status_code}: {response.text}")

shutil.unpack_archive("eurosat.zip", "data/")

if os.path.exists("eurosat.zip"):
    os.remove("eurosat.zip")

os.makedirs("data/images/",exist_ok=True)
os.makedirs("data/labels/", exist_ok=True)

classes = os.listdir("data/2750")

df = pd.DataFrame(columns = ["IMAGE_NAME"]+classes)



for cls in classes:
    class_dir = os.path.join("data/2750", cls)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.verify()
            except (IOError, SyntaxError):
                print(f"Imagen corrupta: {path}")
                os.remove(img_path)  # opcional: eliminarla
                continue
            row = {"IMAGE_NAME": img_name}
            for c in classes:
                row[c] = 1 if c == cls else 0
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            shutil.move(img_path,f"data/images/{img_name}")
df.to_csv("data/labels/labels.csv", index=False)

