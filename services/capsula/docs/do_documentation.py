
"""Execute this script to generate the documentation structure."""

import os
import sys
import shutil

import yaml

SRC_PATH = "../src/"
DOCS_PATH = "../docs/"


new_yaml_structure = [
    {"index": "index.md"},
    {"src": []}
]


def delete_previous_structure() -> None:
    """
    Delete src directory tree in docs.

    Returns:
        None.

    """
    if os.path.exists('./src'):
        shutil.rmtree('./src')


def create_structure_folders(folders: list) -> None:
    """
    Copy src structure in docs.

    Args:
        folders: list of folders in src ordered by parent-child relationship

    Returns:
        None.

    """
    folder_path = "./"
    for folder in folders:
        folder_path += folder + "/"
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)


def process_python_file(dir_path: str, py_file_name: str) -> None:
    """
    Create file.md from file.py.

    Args:
        dir_path: path of the file
        py_file_name: file name

    Returns:
        None.

    """
    dir_path = dir_path.replace("../", "")
    folders = dir_path.split("/")

    file_name_without_extension = py_file_name.replace(".py", "")
    md_file_name = file_name_without_extension + ".md"

    create_structure_folders(folders)
    file_path = "./" + dir_path + "/" + md_file_name

    with open(file_path, "w", encoding='UTF8') as file_md:
        file_md.write(f":::{dir_path.replace('/', '.')}.{file_name_without_extension}\n")
        file_md.write("\toptions:\n\t\tfilters: none")


def process_directory(dir_path: str) -> None:
    """
    Iterate directory's tree finding all the python files.

    Args:
        dir_path: directory from which we want to search the files

    Returns:
        None.

    """
    for element in os.listdir(dir_path):
        path_f = os.path.join(dir_path, element)
        if element.startswith("__"):
            continue
        if os.path.isdir(path_f) and path_f.startswith("__"):
            continue

        if os.path.isdir(path_f):
            process_directory(path_f)
        elif element.endswith(".py"):
            process_python_file(dir_path, element)


def create_nav_yaml(dir_path: str, nav: list) -> None:
    """
    Create dict of tree to navigate in documentation.

    Args:
        dir_path: Path from which we want the dictionary
        nav: list of current directory

    Returns:
        None.

    """
    key = list(nav[-1].keys())[0]
    for element in os.listdir(dir_path):
        path_f = os.path.join(dir_path, element)
        if os.path.isdir(path_f) and path_f.startswith("__"):
            continue
        if os.path.isdir(path_f):
            child_folder = path_f.split('/')[-1]
            new_dict = [{child_folder: []}]
            nav[-1][key].append(new_dict[-1])
            create_nav_yaml(path_f, new_dict)
        elif element.endswith(".md"):
            file_path = dir_path.replace(DOCS_PATH, '') + '/' + element
            nav[-1][key].append(file_path)


if os.getcwd().split("/")[-1] != "docs":
    print("Debes ejecutar el código desde la carpeta de docs.")
    sys.exit()

delete_previous_structure()
process_directory(SRC_PATH)
create_nav_yaml(DOCS_PATH+'src', new_yaml_structure)

with open("../mkdocs.yml", "r", encoding='UTF8') as file:
    yaml_dict = yaml.safe_load(file)
    yaml_dict['nav'] = new_yaml_structure


class WriteLineBreakDumper(yaml.SafeDumper):
    """YAML Dumper to add line breaks between top level objects in YAML file."""

    def write_line_break(self, data=None):
        """
        Add a line break between blocks.

        Args:
            data: Data that we want to write.

        Returns:
            None.

        """
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


with open("../mkdocs.yml", "w", encoding='UTF8') as file:
    yaml.dump(yaml_dict, file, Dumper=WriteLineBreakDumper)
