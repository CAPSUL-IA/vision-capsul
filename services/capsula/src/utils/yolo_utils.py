import json
import os
import shutil
import yaml
import numpy as np

from collections import Counter
from matplotlib import pyplot
from skmultilearn.model_selection import iterative_train_test_split
from config.config import LOGS_PATH


def create_dataset_yaml(path_to_dataset, num_classes, class_names, yaml_file_path):
    """
    Creates a YAML file for configuring a dataset for YOLO training, including paths,
    data augmentation settings, and class information.

    Args:
        path_to_dataset (str): Path to the base directory of the dataset.
        num_classes (int): Number of classes in the dataset.
        class_names (list of str): List containing the names of the classes.
        yaml_file_path (str): Path where the YAML file will be saved.
    """

    dataset_config = {
        'path': path_to_dataset,
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names,
    }

    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(dataset_config, yaml_file, default_flow_style=False, sort_keys=False)


def copy_files(file_list, base_dir, source_dir, target_dir):
    """
    Copies image files and their corresponding label files from source directories
    to target directories.

    Args:
        file_list (list of str): A list of filenames (without extensions) to look for
            in the source directory.
        base_dir (str): The base directory path which contains a 'labels' subdirectory
            for label files.
        source_dir (str): The directory path where image files are located.
        target_dir (str): The directory path where the image files should be copied to.
            Assumes a parallel directory structure for 'images' and 'labels'.

    Note:
        Supported image extensions are '.jpg', '.jpeg', '.png', '.tiff', '.bmp', and
        '.gif'. The function checks for the existence of an image file by appending
        these extensions to the base filenames provided in `file_list`.

    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']
    for filename in file_list:
        base_filename = os.path.splitext(filename)[0]
        img_filename = None
        for ext in image_extensions:
            if os.path.exists(os.path.join(source_dir, base_filename + ext)):
                img_filename = base_filename + ext
                break
        shutil.copy(os.path.join(source_dir, img_filename), target_dir)
        shutil.copy(os.path.join(base_dir, 'labels', filename),
                    target_dir.replace('images', 'labels'))


def create_stratified_yolo_dataset(base_dir, label2id: dict = {}, stratify_ratio=0.2):
    """
    Creates a stratified dataset for YOLO training and validation,
    considering multiple labels in label files. It organizes images
    and labels into a structure expected by YOLO and splits them
    into training and validation sets, ensuring a representative
    class distribution.
    Args:
        base_dir (str): The base directory path where 'images/', 'labels/',
                        and 'label2id.json' are located. This function will
                        create a new subdirectory 'final_data/' here.
        label2id (dict): label encoding dictionary
        stratify_ratio (float): The proportion of the dataset to
                                include in the validation split.

    Returns:
        None
    """

    # Prepare directories for the final structured dataset
    final_data_dir = os.path.join(base_dir, 'final_data')
    train_img_dir = os.path.join(final_data_dir, 'images/train')
    val_img_dir = os.path.join(final_data_dir, 'images/val')
    train_label_dir = os.path.join(final_data_dir, 'labels/train')
    val_label_dir = os.path.join(final_data_dir, 'labels/val')
    for directory in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(directory, exist_ok=True)

    label_files = [f for f in os.listdir(os.path.join(base_dir, 'labels')) if f.endswith('.txt')]
    label_frequencies = Counter()
    stratify_labels = []

    for label_file in label_files:
        with open(os.path.join(base_dir, 'labels', label_file)) as f:
            lines = f.readlines()
            labels_in_file = np.zeros(len(label2id))
            unique_labels_in_file = set()
            for line in lines:
                class_id = int(line.split()[0])
                labels_in_file[class_id] = 1
                unique_labels_in_file.add(class_id)
            stratify_labels.append(labels_in_file)
            label_frequencies.update(unique_labels_in_file)
    stratify_labels = np.array(stratify_labels)
    # Split the dataset
    x = np.array(label_files).reshape(-1, 1)
    y = stratify_labels
    x_train, _, x_val, _ = iterative_train_test_split(x, y, test_size=stratify_ratio)
    # Flatten X arrays for file operations
    x_train = x_train.flatten()
    x_val = x_val.flatten()
    # Copy files to the structured directories
    copy_files(x_train, base_dir, os.path.join(base_dir, 'images'), train_img_dir)
    copy_files(x_val, base_dir, os.path.join(base_dir, 'images'), val_img_dir)
    _, frequencies = zip(*label_frequencies.items())
    create_detection_histogram(label2id, frequencies)


def get_label2id(base_dir):
    """
    Return label encoding of the dataset
    Args:
        base_dir (str): The base directory path where 'images/', 'labels/',
                        and 'label2id.json' are located. This function will
                        create a new subdirectory 'final_data/' here.
    Returns:
        label encoding dictionary
    """
    with open(os.path.join(base_dir, 'label2id.json')) as f:
        label2id = json.load(f)
    return label2id


def create_detection_histogram(label2_to_id: dict, frequencies: list):
    """
    Function to create a histogram with label
    distribution of object detection problem. It counts
    the frequency of unique label in all processed label files.
    Args:
        label2_to_id: List whit label names
        frequencies: List wit frequencies of each label
    """
    pyplot.figure(figsize=(10, 6))
    bars = pyplot.bar(range(len(label2_to_id)), [frequencies[label2_to_id[label]] for label in label2_to_id],
                      tick_label=list(label2_to_id.keys()))
    pyplot.xlabel('Label')
    pyplot.ylabel('Frequency')
    pyplot.title('Label Distribution')
    pyplot.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        pyplot.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, yval, ha='center', va='bottom')
    pyplot.tight_layout()
    plots_path = os.path.join(LOGS_PATH, 'plots/')
    
    # Create the logs/plots folder if it does not exist
    os.makedirs(plots_path, exist_ok=True)
    pyplot.savefig(os.path.join(plots_path, "label_distribution.jpg"))
