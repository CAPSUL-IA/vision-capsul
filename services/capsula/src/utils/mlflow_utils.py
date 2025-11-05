
"""Module with all the MLFlow settings functions."""

import os
import json
import shutil

import boto3
import mlflow
import mlflow.entities
from mlflow.tracking import MlflowClient
from botocore.exceptions import ClientError

from config.mlflow_config import (
    SERVER_URL, TMP_PATH, CREDENTIALS_PATH, CREDENTIALS_FILENAME, S3_BUCKET, S3_CREDENTIALS_FILENAME
)


def _set_mlflow_env_variable() -> None:
    """
    Set MLFlow environment variables,

    Returns:
        None.

    """
    with open(f"{CREDENTIALS_PATH}{CREDENTIALS_FILENAME}", 'r') as f_read:
        credentials = json.load(f_read)

    for key, value in credentials.items():
        os.environ[key] = value


def download_credentials_from_s3() -> None:
    """
    Download the mlflow credentials file from Amazon S3.

    Returns:
        None.

    """
    s3 = boto3.client('s3')

    try:
        s3.download_file(S3_BUCKET, S3_CREDENTIALS_FILENAME, CREDENTIALS_FILENAME)
    except Exception as e:
        print(f"Error downloading file: {e}")


def _create_credentials() -> None:
    """
    Create credentials in default path.

    Returns:
        None.

    """
    if not os.path.exists(f"{CREDENTIALS_PATH}"):
        os.makedirs(CREDENTIALS_PATH, exist_ok=True)

    download_credentials_from_s3()

    shutil.move(CREDENTIALS_FILENAME, f"{CREDENTIALS_PATH}{CREDENTIALS_FILENAME}")


def _check_credentials() -> None:
    """
    Check if the credentials are stored.

    Returns:
        None.

    """
    if (
        not os.path.exists(f"{CREDENTIALS_PATH}") or
        not os.path.exists(f"{CREDENTIALS_PATH}{CREDENTIALS_FILENAME}")
    ):
        _create_credentials()


def set_mlflow_config(prefix_exp_name: str) -> mlflow.entities.Experiment:
    """
    Gets or creates the experiment given and returns it.

    Args:
        prefix_exp_name: String with the name of the experiment.

    Returns:
        The mlflow experiment object.

    """
    _check_credentials()

    _set_mlflow_env_variable()

    # End any previous active run
    mlflow.end_run()

    # Experiments will be stored in /home/USER_NAME/mlruns
    experiment_name = prefix_exp_name

    # Where to save all registered metadata
    mlflow.set_tracking_uri(f"{SERVER_URL}")

    client = MlflowClient()

    # Create new if experiment does not exist or has been deleted
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        client.create_experiment(name=experiment_name)
    elif experiment.lifecycle_stage == 'deleted':
        client.restore_experiment(experiment.experiment_id)

    experiment = client.get_experiment_by_name(experiment_name)
    print(f"Training metadata will be logged in {experiment.name} MlFlow project")
    return experiment


def clear_tmp() -> None:
    """
    Delete all files and directories inside TMP_PATH.

    Returns:
        None.

    """
    if not os.path.exists(TMP_PATH):
        os.mkdir(TMP_PATH)
    for filename in os.listdir(TMP_PATH):
        file_path = os.path.join(TMP_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exc:
            print(f'Failed to delete {file_path}. Reason: {exc}')


def upload_file(file_name: str, s3_name: str, bucket_path: str) -> bool:
    """
    Upload file to S3.

    Args:
        file_name: Path of the file we want to upload
        bucket_path: Path with the S3 bucket

    Returns:
        Boolean specifying if file was uploaded successfully or not.

    """
    # object_name = os.path.basename(file_name)

    uri_elements = bucket_path.replace("s3://", "").split("/")
    bucket = uri_elements[0]
    s3_path = "/".join(uri_elements[1:])

    object_name = s3_path + s3_name

    print(f"Local file {file_name} -> S3: {bucket} - {object_name}")

    # Upload the file
    s3_client = boto3.client('s3', region_name='eu-west-1')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True


def upload_folder(folder_path: str, relative_path: str, bucket_path: str):
    for object_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, object_name)
        if not os.path.isdir(file_path):
            upload_file(file_path, relative_path + "/" + object_name, bucket_path)
        else:
            upload_folder(file_path, relative_path + "/" + object_name, bucket_path)


def upload_artifacts(model_uri: str) -> None:
    """
    Uploads all the files from TMP_PATH to MLFlow run.

    Args:
        model_uri: URI to run made

    Returns:
        None.

    """
    for object_name in os.listdir(TMP_PATH):
        file_path = os.path.join(TMP_PATH, object_name)
        if not os.path.isdir(file_path):
            upload_file(file_path, object_name, model_uri)
        else:
            upload_folder(file_path, object_name, model_uri)
