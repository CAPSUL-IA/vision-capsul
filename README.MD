# Capsul-IA vision

A brief tutorial on how to use.

## Installation

In order to install the Capsul-IA project, you need to have to install the following software:

- [Docker](https://www.docker.com/) latest version
- [Visual Studio Code](https://code.visualstudio.com/) (optional)

## Usage

To use the Capsul-IA project, you only need to run the following command in the root directory of the project:

```bash
bash scripts/run-dev.sh
```
## Develop
Once you have been executed the `./scripts/run-dev.sh` command, you can open the project in Visual Studio Code. Just click on the left bottom corner of the window and select the `Attach to running container` option. This will open a new window with the project files and the development environment running as it was in your local machine. You will need to install the `Remote - Containers` extension in Visual Studio Code to use this feature.

Using Dev containers allows you to develop in the same environment as the production environment. This means that you will have the same services and configurations as the production environment. This will help you to avoid the "It works on my machine" problem. It also reduces the time to set up the development environment if you need to incorporate new developers to the project.

The first time you open the Dev container in Visual Studio Code, it will prompt you to open the folder in the container. Just click on the `Open folder` button and use  `/app` as the folder path.

Feel free to install any extension you need in the Dev container. The extensions will be installed in the Dev container and not in your local machine. And will be available the next time you open the capsule.

## Data for classification / detection
If you want to change where your data is saved or use your own data, just update this line in ```environments/develop/docker-compose.yml``` with the path you prefer:

```- your/data/path/:/app/data```

