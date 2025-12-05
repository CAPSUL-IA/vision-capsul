# CapsulIA
Demo para la capsula de visión.
## Formato de los datos

En este apartado se va a exponer la forma en la que se deben preparar los datos para
realizar el entrenamiento de un modelo tanto en clasifiación como en detección.

### Formato de los datos para clasificación
Los datos deben organizarse en dos carpetas principales:

* **images**: Esta carpeta debe contener todas las imágenes que se utilizarán para el entrenamiento o la evaluación del modelo.
* **labels**: Dentro de esta carpeta debe incluirse un archivo en formato CSV (labels.csv) que organiza las etiquetas correspondientes a cada imagen.

Estas dos carpetas deberán copiarse dentro de la carpeta `data/` de este repositorio.
Por tanto, la estructura de carpetas esperada será la siguiente:
~~~
capsula
    |--data
        |--images
            |--image1.jpg
            |--image2.jpg
            |--image3.jpg
            ....
        |--labels
            |--labels.csv
~~~
#### Estructura del archivo labels.csv

El archivo CSV dentro de la carpeta labels puede tener dos formatos de columnas. Primero:
* IMAGE_NAME: El nombre del archivo de imagen correspondiente. Este nombre debe incluir la extensión del archivo (por ejemplo, .jpg, .png).
* LABELS: Las etiquetas asignadas a cada imagen.

Segundo:
* IMAGE_NAME: El nombre del archivo de imagen correspondiente. Este nombre debe incluir la extensión del archivo (por ejemplo, .jpg, .png).
* Una columna por clase, siendo que el valor de la celda será 1, si la imagen es de la clase, o 0 en caso contrario.



##### Ejemplo labels.csv para problema multiclase
Suponiendo un problema multiclase con dos etiquetas, el formato
del fichero labels.csv esperado será:

~~~
IMAGE_NAME      LABELS
image1.jpg       uno
image2.jpg       dos
...
~~~
~~~
IMAGE_NAME      uno	dos
image1.jpg       1	 0
image2.jpg       0	 1
...
~~~

##### Ejemplo labels.csv para problema multilabel
Suponiendo un problema multilabel con dos etiquetas, el formato
del fichero labels.csv esperado será:

~~~
IMAGE_NAME      LABELS
image1.jpg       uno,dos
image2.jpg       dos
...
~~~
~~~
IMAGE_NAME      uno	dos
image1.jpg       1	 1
image2.jpg       0	 1
...
~~~


### Formato de los datos para detección

* **Carpeta images**: Esta carpeta debe contener todas las imágenes que se utilizarán para el entrenamiento del modelo.

* **Carpeta labels**: Dentro de esta carpeta debe incluirse un archivo txt para cada una de las imágenes con las etiquetas en formato YOLO. Cada txt tiene que tener el mismo nombre que la imagen de la que contiene sus anotaciones.

* **Fichero label2id.json**: Este fichero sirve para mapear el nombre de las clases con su id asociado en las etiquetas de los ficheros txt que se encuentran en la carpeta labels.

Estas dos carpetas y el fichero json deberán copiarse dentro de la carpeta `data/` de este repositorio. Por tanto, la estructura de carpetas esperada será la siguiente:

~~~
capsula
    |--data
        |--images
            |--image1.jpg
            |--image2.jpg
            |--image3.jpg
            ....
        |--labels
            |--image1.txt
            |--image2.txt
            |--image3.txt
            ....
        |-- label2id.json
~~~

Existe un parámetro configurable (`Stratify`) que expresa si se quiere estratificar el dataset o no. Esto es, en caso de que el parámetro tenga un valor positivo, cogerán todas las imágenes y sus anotaciones y se crearán dos subconjuntos (train, val) nuevos donde cada uno tiene una proporción de representación de las diferentes clases igual. Tras este proceso, la estructura final de la carpeta `data/` quedará de la siguiente manera:

~~~
capsula
    |--data
        |--final_data
            |--images
                |--train
                    |--image1.jpg
                    |--image2.jpg
                    |--...
                |--val
                    |--image3.jpg
                    |--image4.jpg
                    |--...
            |--labels
                |--train
                    |--image1.txt
                    |--image2.txt
                    |--...
                |--val
                    |--image3.txt
                    |--image4.txt
                    |--...
        |--images
            |--image1.jpg
            |--image2.jpg
            |--image3.jpg
            ....
        |--labels
            |--image1.txt
            |--image2.txt
            |--image3.txt
            ....
        |-- label2id.json
~~~

En caso de que se disponga de datos ya estratificados -que ya estén divididos en estos subconjuntos, este parámetro se configurará a `No` y la estructura de la carpeta `/data` será como sigue:

~~~
capsula
    |--data
        |--images
            |--train
                |--image1.jpg
                |--image2.jpg
                |--...
            |--val
                |--image3.jpg
                |--image4.jpg
                |--...
        |--labels
            |--train
                |--image1.txt
                |--image2.txt
                |--...
            |--val
                |--image3.txt
                |--image4.txt
                |--...
        |-- label2id.json
~~~

### Transformación de etiquetas a formato YOLO
En caso de tener etiquetas en un formato distinto al esperado, para poder tener las etiquetas en formato YOLO de manera más simple, se podrían transformar fácilmente con:

```bash
python3 scripts/transform_labels.py --data data/ --from coco --to yolo
```

Este script permite la transformación de varios formatos de etiquetas a YOLO, o el cambio de YOLO a alguno de estos. 

### Ejemplo contenido label2id.json

Suponiendo que en un caso de uso se quieren detectar en las imágenes plátanos, manzanas y naranjas, un posible contenido de label2id.json podría ser:

~~~
{"apple": 0, "banana": 1, "orange": 2}
~~~

### Ejemplo contenido ficheros txt dentro de carpeta labels.

En este formato cada línea se corresponde a un objeto en la imagen y contiene
información como la categoría del objeto, el cuadro delimitador del objeto.
El cuadro delimitador se representa como **x, y, width, height**,
donde `x` e `y` son las coordenadas del punto medio del cuadro delimitador,
y `width` y `height` son el ancho y el alto del cuadro delimitador, respectivamente.
Estos valores se normalizan para tener una escala de 1, es decir, se dividen por el
ancho y el alto de la imagen, respectivamente. 

~~~
id_categoria1 x_norm y_norm width_norm heihgt_norm
id_categoria2 x_norm y_norm width_norm heihgt_norm
id_categoria3 x_norm y_norm width_norm heihgt_norm
....
~~~

Así pues, suponiendo una imagen, image1.jpg que tenga una manzana y una naranja etiquetados, el fichero image1.txt
asociado dentro de la carpeta labels podría ser:

~~~
0 0.34921875 0.30280046674445743 0.3015625 0.2602100350058343
2 0.53984375 0.5746791131855309 0.8515625 0.2952158693115519
~~~
En el que el primer elemento se corresponde con el id de la clase, el segundo con la coordenada x normalizada,
el tercero con la coordenada y normalizada, y los otros dos con width y height normalizados.

## Estructura de ` run.cfg`
El archivo está dividido en varias secciones, cada una destinada a un aspecto específico de la configuración.

Los valores **booleanos** aceptan los valores `1 | yes | true | on | 0 | no | false | off`.

### [Task]
* **Type**: Indica la tarea a realizar, pudiendo ser esta object_detection o classification.

### [Images]
* **Size**: Define el tamaño de las imágenes (altura y anchura en píxeles) que se utilizarán para el
entrenamiento y la evaluación. Si no se especifica, el valor por defecto está establecido en 256.
* **Fraction**: Define la partición de datos que se va a usar para el entrenamiento, siendo 1.0 el dataset completo y 0.1 un 10%.

### [Model]

* **Name**: Especifica el modelo de red neuronal utilizado para la clasificación o la detección.
Para la cápsula encargada de la clasificación imágenes se ha utilizado la librería [Timm](https://huggingface.co/docs/timm/quickstart), que permite el uso de modelos de clasificación se debe a su amplia colección de arquitecturas, tanto clásicas como de última generación, que han demostrado ser efectivas.

Para conocer qué modelos con pesos preentrenados se encuentran disponibles, se puede consultar la documentación de [Timm](https://huggingface.co/docs/timm/models) o bien, ejecutar la siguiente instrucción
en python:

~~~
import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
print(model_names)
~~~

Para la cápsula encargada de la detección de imágenes se han utilizados los modelos [YOLO](https://docs.ultralytics.com/es/models/).

### [Preprocess]
* **Stratify** (booleano): Especifica si los datos deben de ser estratificados o no.
* **MultiLabel** (booleano): Especifica si la tarea de clasificación se trata de un problema multietiqueta o no.

### [ModelParams]
* **Pretrained** (booleano): En caso de que sea configurable como en el caso de los modelos de la biblioteca `timm`, este parámetro especifica si utilizar el modelo con pesos preentrenados o no.
* **Overwrite** (booleano): Especifica si, al ejecutar un nuevo entrenamiento, conservar un modelo existente anterior o sobreescribirlo.
* **Amp** (booleano): Especifica si, al entrenar, se usa el modo de precisión mixta, lo que permite entrenamiento más ligeros y más rápidos.

### [Quantization]
* **Quantize**: Permite cuantizar el modelo a **float16** o **int8** para la exportación.

### [LearningParams]

* **Batch**: Tamaño de batch utilizado en el entrenamiento.
Si no se especifica, el valor por defecto está establecido en 32.

* **Epoch**: Número de epochs a completar en el entrenamiento.
Si no se especifica, el valor por defecto está establecido en 25.

* **EvalPartition**: Fracción del conjunto de datos a utilizar para la evaluación (validación). Si no se especifica, el valor por defecto está establecido en 0.2.

* **Optim**: Algoritmo de optimización utilizado. Se puede elegir entre Adam, AdamW y SGD.
Si no se especifica, el valor por defecto está establecido en Adam.

* **LearningRate**: Tasa de aprendizaje inicial.
Si no se especifica, el valor por defecto está establecido en 0.001.

* **Scheduler**: Programador de la tasa de aprendizaje. Se puede elegir entre Cosine y Plateau. Si no se especifica, el valor por defecto está establecido en Cosine.

* **Patience**: Número de epochs a esperar para la mejora antes de detener el entrenamiento.
Si no se especifica, el valor por defecto está establecido en None (es decir, no habrá early stopping).

## Entrenamiento
Para iniciar el proceso de entrenamiento tanto para clasificación como detección,
se debe ejecutar el siguiente comando en la terminal:

~~~
python3 train.py
~~~

Finalmente, al concluir el entrenamiento, los modelos resultantes se encontrarán en la carpeta models/

## Inferencia
El script inference.py se ejecuta desde la línea de comandos y acepta varios argumentos que
permiten especificar la configuración de la inferencia. A continuación, se describen los argumentos
soportados:

- --img_folder: Ruta obligatoria al directorio que contiene las imágenes sobre las que se realizará la inferencia.

- --model_path: Ruta opcional al punto de control del modelo. Por defecto, se utiliza models/classificator/model.onnx que también es la salida de los modelos de clasificación.

- --task: Tipo de tarea a realizar, siendo la opción **c** clasificación y **d** detección. Por defecto, se utiliza c.

Para ejecutar inferencia con el script inference.py:

~~~
python inference.py --img_folder /ruta/a/tu/carpeta/de/imagenes --model_path ruta/a/salida/onnx --task d
~~~

## Preparación de las imágenes
Se cuenta con dos scripts que permiten la descarga y preparación de los conjuntos de imágenes CrowdHuman(detección) y EuroSat(clasificación). Para usarlos:
```
python3 scripts/download_crowdhuman.py
python3 scripts/download_eurosat.py
```
