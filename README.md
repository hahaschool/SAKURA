# SAKURA: Single-cell data Analysis with Knowledge inputs from User using Regularized Autoencoders

<br />

<img src="https://github.com/VitoChan1/SAKURA/blob/main/Icon.png" align="right" alt="SAKURA" width="100" />

Single-cell data Analysis with Knowledge inputs from User using Regularized Autoencoders (SAKURA) is a knowledge-guided dimensionality reduction framework. 
It focuses on the task of producing an embedding (i.e., a low-dimensional representation) of scRNA-seq or scATAC-seq data, to be guided by a large variety of knowledge inputs related to genes and genomic regions. 

<img width=100% src="https://github.com/VitoChan1/SAKURA/blob/main/docs/source/static/FigOverallDesign.png"/>

<p align="center">
  Read our paper at <a href="https://www.biorxiv.org/content/10.1101/2025.10.01.679835v1" target="_blank">A knowledge-guided approach to recovering important rare signals from high-dimensional single-cell data</a>.
</p>

<p align="center">
  For detailed documentation and usage tutorials, please refer to <a href="https://yip-sakura.readthedocs.io/en/latest/" target="_blank">https://yip-sakura.readthedocs.io/en/latest/</a>.
</p>

<p align="center">
  For documentation source and related data, you can visit the <a href="https://github.com/VitoChan1/SAKURA" target="_blank">Documentation Repository</a>.
</p>

<!-- 
In the [user guide](https://), we provide an overview of each model.
All model implementations have a high-level API that interacts with
[scanpy](http://scanpy.readthedocs.io/) and includes standard save/load functions, GPU acceleration, etc.
--> 

## Directory structure

```
.
├── sakura/                 # Main Python package
│   ├── dataset             # Input dataset handling classes
│   ├── model_controllers   # Model workflow controllers 
│   ├── models              # Model components and architectures 
│   ├── utils               # Utilities: dataset splitter, distance computation, etc.
│   ├── sakuraAE.py         # Generic SAKURA pipeline class
├── pyproject.toml          # Python package metadata
├── poetry.lock             # Poetry locked dependencies for consistent installs
├── requirements.txt        # Projen: Main dependencies for production
├── requirements-dev.txt    # Projen: Additional dependencies for development
├── Icon.png
├── LICENSE
└── README.md
```

## Getting Started  

Clone the repository to your local directory. The `SAKURA` package can be installed via pip directly:

```bash
# in ~/.../SAKURA/
pip install .
```
> Installing within a
> [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
> is recommended.

## Basic Usage
<!--### Training a SAKURA model-->
The `./sakura/sakuraAE.py` script handles the different operations of the generic SAKURA framework. Explore the framework using the following command and optional arguments:

```sh
# in ~/.../SAKURA/
python -m sakura -h

...
usage: SAKURA [-h] [-c CONFIG] [-v VERBOSE] [-s SUPPRESS_TRAIN] [-r RESUME] [-i INFERENCE] [-y INFERENCE_STORY] [-x SUPPRESS_TENSORBOARDX]
              [-e EXTERNAL_MODULE] [-E EXTERNAL_MODULE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        model config JSON path
  -v VERBOSE, --verbose VERBOSE
                        verbose console outputs
  -s SUPPRESS_TRAIN, --suppress_train SUPPRESS_TRAIN
                        suppress model training, only setup dataset and model
  -r RESUME, --resume RESUME
                        resume training process from saved checkpoint file
  -i INFERENCE, --inference INFERENCE
                        perform inference from saved checkpoint file containing models
  -y INFERENCE_STORY, --inference_story INFERENCE_STORY
                        story file of inference
  -x SUPPRESS_TENSORBOARDX, --suppress_tensorboardX SUPPRESS_TENSORBOARDX
                        suppress Logger to initiate tensorboardX (to prevent flushing logs)
  -e EXTERNAL_MODULE, --external_module EXTERNAL_MODULE
                        insert modules from external (pretrained) models
  -E EXTERNAL_MODULE_PATH, --external_module_path EXTERNAL_MODULE_PATH
                        path of external model config
```

By default, the SAKURA model training workflow involves loading all the data from `input file paths` and then splitting the preprocessed data into training and testing sets, with detail configuration about `how to split/store the datasets`. 
The SAKURA autoencoder model will be built based on the parameters including `the number of encoder/decoder neurons, the dimension of latent space, the loss function, the regularization`, and so on.
To facilitate model training and testing, users also specify `optimizer settings, checkpoint and test segment, logging and result dumping settings`. 
All of these relevant settings can be conveniently specified in the model JSON configuration file, allowing them to be easily passed into SAKURA as a params dictionary instead of using an argument parser. 

<!--### Performing classification with trained netAE model
After training, one may want to use a classifier on the embedded space to test its classification accuracy. The `inference.py` script deals with comparing classification accuracy of netAE with other baseline models when using KNN and logistic regression, two simple classifiers. To start, make sure netAE, AE (the unsuperivsed counterpart), scVI, PCA, and ZIFA are trained and have their embedded spaces located in `MODEL_PATH`. Then simply pass in `--data-path`, `--model-path`, `--lab-size`, and `--dataset`. Additionally, to ensure that the labeled set used in training netAE is the same as here, make sure that you pass in the same seed `--seed` here as when training netAE.
-->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

