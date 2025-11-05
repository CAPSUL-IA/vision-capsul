import argparse
import os
import cv2

def pascal_voc_to_yolo(bbox, video_resolution):
    video_w, video_h = video_resolution
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    xc = x_min + 0.5 * width
    yc = y_min + 0.5 * height
    return xc / video_w, yc / video_h, width / video_w, height / video_h

def coco_to_yolo(bbox, video_resolution):
    video_w, video_h = video_resolution
    x_min, y_min, width, height = bbox
    xc = x_min + 0.5 * width
    yc = y_min + 0.5 * height
    return xc / video_w, yc / video_h, width / video_w, height / video_h

def cxcywh_to_yolo(bbox, video_resolution):
    video_w, video_h = video_resolution
    xc, yc, width, height = bbox
    return xc / video_w, yc / video_h, width / video_w, height / video_h

def albumentations_to_yolo(bbox, video_resolution):
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
    w_norm = x_max_norm - x_min_norm
    h_norm = y_max_norm - y_min_norm
    x_norm = x_min_norm + 0.5 * w_norm
    y_norm = y_min_norm + 0.5 * h_norm
    return x_norm, y_norm, w_norm, h_norm

def yolo_to_pascal_voc(bbox, video_resolution):
    video_w, video_h = video_resolution
    x_norm, y_norm, w_norm, h_norm = bbox
    xc, yc = x_norm * video_w, y_norm * video_h
    width, height = w_norm * video_w, h_norm * video_h
    x_min = xc - 0.5 * width
    y_min = yc - 0.5 * height
    x_max = xc + 0.5 * width
    y_max = yc + 0.5 * height
    return x_min, y_min, x_max, y_max

def yolo_to_coco(bbox, video_resolution):
    video_w, video_h = video_resolution
    x_norm, y_norm, w_norm, h_norm = bbox
    xc, yc = x_norm * video_w, y_norm * video_h
    width, height = w_norm * video_w, h_norm * video_h
    x_min = xc - 0.5 * width
    y_min = yc - 0.5 * height
    return x_min, y_min, width, height

def yolo_to_cxcywh(bbox, video_resolution):
    video_w, video_h = video_resolution
    x_norm, y_norm, w_norm, h_norm = bbox
    xc = x_norm * video_w
    yc = y_norm * video_h
    width = w_norm * video_w
    height = h_norm * video_h
    return xc, yc, width, height

def yolo_to_albumentations(bbox, video_resolution):
    x_norm, y_norm, w_norm, h_norm = bbox
    x_min = x_norm - 0.5 * w_norm
    y_min = y_norm - 0.5 * h_norm
    x_max = x_norm + 0.5 * w_norm
    y_max = y_norm + 0.5 * h_norm
    return x_min, y_min, x_max, y_max

TO_YOLO_FUNCS = {
    "pascal": pascal_voc_to_yolo,
    "coco": coco_to_yolo,
    "cxcywh": cxcywh_to_yolo,
    "albumentations": albumentations_to_yolo,
    "yolo": lambda bbox, res=None: bbox  
}

FROM_YOLO_FUNCS = {
    "pascal": yolo_to_pascal_voc,
    "coco": yolo_to_coco,
    "cxcywh": yolo_to_cxcywh,
    "albumentations": yolo_to_albumentations,
    "yolo": lambda bbox, res=None: bbox  
}

def convert_bbox(bbox, from_fmt, to_fmt, resolution): 
    yolo_bbox = TO_YOLO_FUNCS[from_fmt](bbox, resolution)
    return FROM_YOLO_FUNCS[to_fmt](yolo_bbox, resolution)

def convert_label_file(label_path, image_path, from_fmt, to_fmt):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se ha podido abrir la imagen: {image_path}")
        return

    height, width = img.shape[:2]
    video_resolution = (width, height)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        bbox = list(map(float, parts[1:5]))

        converted_bbox = convert_bbox(bbox, from_fmt, to_fmt, video_resolution)
        converted_str = " ".join(f"{v}" for v in converted_bbox)
        converted_lines.append(f"{cls} {converted_str}\n")

    with open(label_path, 'w') as f:
        f.writelines(converted_lines)


def process_dataset(data_dir, from_fmt, to_fmt):
    images_root = os.path.join(data_dir, "images")
    labels_root = os.path.join(data_dir, "labels")

    subsets = ["train", "val", "test"]

    for subset in subsets:
        image_dir = os.path.join(images_root, subset)
        label_dir = os.path.join(labels_root, subset)

        if not os.path.exists(image_dir):
            print(f"No existe el directorio {image_dir}.")
            continue

        for file in os.listdir(image_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(image_dir, file)
            base = os.path.splitext(file)[0]
            label_path = os.path.join(label_dir, f"{base}.txt")

            if not os.path.exists(label_path):
                print(f"No se han encontrado las etiquetas para {file}")
                continue

            convert_label_file(label_path, image_path, from_fmt, to_fmt)

def main():
    parser = argparse.ArgumentParser(description="Conversor entre formatos de etiquetas.")
    parser.add_argument("--data", required=True, default="data/",  help="Ruta al directorio base (por ejemplo, data/)")
    parser.add_argument("--from", dest="from_fmt", required=True,
                        choices=["pascal", "coco", "cxcywh", "albumentations", "yolo"],
                        help="Formato actual de las labels")
    parser.add_argument("--to", dest="to_fmt", default="yolo",
                        choices=["pascal", "coco", "cxcywh", "albumentations", "yolo"],
                        help="Formato destino (por defecto YOLO)")
    args = parser.parse_args()

    process_dataset(args.data, args.from_fmt, args.to_fmt)

if __name__ == "__main__":
    main()

