
Ref Repository: https://github.com/google/sg2im


# Object Alignment

Folder: Object_Alignment

## LayoutGAN for bbox experiments

First, download the trasformed point layout representation of MNIST dataset from
https://drive.google.com/file/d/1R1iRZxADR_RcDsuR4gyStyLAo7i5LRAH/view?usp=sharing,
and put it under ./data directory.

To train a model with downloaded dataset:
$ bash ./experiments/scripts/train_mnist.sh

In order to run the codes for each experiment, you need to save this folder in your Google Drive. After adding in your drive, you can simply open the notebooks in Google Colab and run the cells.

You need to mount your drive on Google Colab before running the cells in notebook so that files could be accessed without any error.

## Dataset Preparation

(NOTE: We have already downloaded and processed the data and saved it as numpy array, so you can skip the steps below)

1. Use [PubLayNet dataset](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet.html). Download `labels.tar.gz` and decompress it.
2. Run `python preprocess_doc.py` for preprocessing dataset.

## Prerequisites

-   Python 2.7
-   Tensorflow 1.2.0
-   [COCO API](https://github.com/cocodataset/cocoapi)


## 


# Image Generation
Ref for this readme: https://github.com/google/sg2im/blob/master/TRAINING.md


## Step 1: Install COCO API
To train new models you will need to install the [COCO Python API](https://github.com/cocodataset/cocoapi). Unfortunately installing this package via pip often leads to build errors, but you can install it from source like this:

```bash
cd ~
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
python setup.py install
```

## Step 2: Preparing the data
### COCO
Run the following script to download and unpack the relevant parts of the COCO dataset:

```bash
bash scripts/download_coco.sh
```

This will create the directory `datasets/coco` and will download about 21 GB of data to this directory; after unpacking it will take about 60 GB of disk space.


## Step 3: Train a model

Now you can train a new model by running the script:

```bash
python scripts/train.py
```
To train the model:
python scripts/sample.py --batch_size [BATCH_SIZE] --num_epochs [NUM_EPOCHS] --learning_rate [0.0001] --coco_train_image_dir [COCO_IMG_DIR] --coco_val_image_dir [COCO_VAL_DIR] --coco_train_instances_json [PATH_TO_TRAIN_INSTANCE_JSON] --coco_train_stuff_json [PATH_TO_TRAIN_STUFF_JSON] --coco_val_instances_json [PATH_TO_VAL_INSTANCE_JSON_FILE] --coco_val_stuff_json [PATH_TO_VAL_STUFF_JSON_FILE]


- `--batch_size`: How many pairs of (scene graph, image) to use in each minibatch during training. Default is 5.
- `--num_epochs`: Number of training iterations. Default is 100.
- `--learning_rate`: Learning rate to use in Adam optimizer for the generator and discriminators; default is 1e-4.

### Dataset options

- `--dataset`: The dataset to use for training;
- `--image_size`: The size of images to generate, as a tuple of integers. Default is `64,64`. This is also the resolution at which scene layouts are predicted.

**COCO options**:
These flags only take effect if `--dataset` is set to `coco`:

- `--coco_train_image_dir`: Directory from which to load COCO training images; 
- `--coco_val_image_dir`: Directory from which to load COCO validation images; 
- `--coco_train_instances_json`: Path to JSON file containing object annotations for the COCO training images; 
- `--coco_train_stuff_json`: Path to JSON file containing stuff annotations for the COCO training images; 
- `--coco_val_instances_json`: Path to JSON file containing object annotations for COCO validation images; 
- `--coco_train_instances_json`: Path to JSON file containing stuff annotations for COCO validation images; 



## Pretrained Models
Download the model weights from https://drive.google.com/file/d/1kw92ceFq6bylYQ4aw1wPxWnUglYvuCuz/view?usp=sharing


## Sample generated images
Following command will generate some images given a set of objects from the validation set of COCO.

To sample some of the generate images:
python scripts/sample.py --output_folder [PATH_TO_STORE_IMAGES] --num_sample_imgs [NUM_OF_IMAGES] --checkpoint_start_from [MODEL_WEIGHTS_PATH]





