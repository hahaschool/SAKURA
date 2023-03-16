<p align="center">
<img src="Icon.png" alt="SAKURA" height="200" >
</p>

# SAKURA: Single-cell data Analysis with Knowledge inputs from User using Regularized Autoencoders

Single-cell data Analysis with Knowledge inputs from User using Regularized Autoencoders (SAKURA) is a knowledge-guided dimensionality reduction framework. 
It focuses on the task of producing an embedding (i.e., a low-dimensional representation) of scRNA-seq or scATAC-seq data, to be guided by a large variety of knowledge inputs related to genes and genomic regions. 

## Analysis of single-cell data

SAKURA is composed of modules for the following types of knowledge inputs:

-   Marker genes
-   Genes about confounding factors
-   Orthologous genes
-   Invariant genes
-   Bulk-sample measurements
-   Regulatory elements
-   and more to explore!

<!-- 
In the [user guide](https://), we provide an overview of each model.
All model implementations have a high-level API that interacts with
[scanpy](http://scanpy.readthedocs.io/) and includes standard save/load functions, GPU acceleration, etc.
-->

## Getting Started  

Clone the repository to your local directory. Then

```
# in ~/.../SAKURA/
python3 .projenrc.py
```

## Data loading

The `./sakura/dataset/rna_count*.py` scripts handle data loading and preprocessing. 
For example, `dataset = rna_count_sparse.SCRNASeqCountData(gene_MM_path,
gene_name_csv_path,
cell_name_csv_path,
pheno_csv_path,
pheno_meta_json_path)` 
loads the gene expression data from `gene_MM_path`, together with gene names and cell names from corresponding csv files to build pandas sparse DataFrame. 
Also, phenotype data frame and phenotype column metadata are loaded from the corresponding csv files. Note that gene expression matrix has shape (ngenes, nsamples), while the phenotype matrix has shape (nsamples, nphenotypes).

<!--To use the data for training, `dataset.load_all()` returns the following:
- `expr`: preprocessed expression matrix as a numpy array
- `lab_full`: labels of all samples
- `labeled_idx`: indices of the randomly selected labeled set
- `unlabeled_idx`: indices of the rest of the samples
- `info`: additional dictionary containing information of the dataset. `info["cell_type"]` is a dictionary that maps each label to the name of the cell type. `info["cell_id"]` contains the cell ID in the original dataset. `info["gene_names"]` contains the gene names of the dataset.

To load a small subset of the samples for testing, call `dataset.load_subset(p)` instead, where `p` specifies the percentage of all samples to load.-->

## Basic Usage
<!--### Training a SAKURA model-->
The `sakura.py` script handles the different operations of the generic SAKURA framework. Explore the framework using the following command and some important argument parsers:
```
python3 sakura.py

-c, --config            # model JSON configuration file path (including details of data loading & splitting, model settings, training & testing, checkpoint & result saving, etc.) 
-s, --suppress_train    # suppress model training, only setup dataset and model
-r, --resume            # resume training process from saved checkpoint file
-i, --inference         # perform inference from saved checkpoint file containing models

## external model
-e, --external_module       # insert modules from external (pretrained) models
-E, --external_module_path  # path of external model config
```

By default, the SAKURA model training workflow involves loading all the data from `input file paths` and then splitting the preprocessed data into training and testing sets, with detail configuration about `how to split/store the datasets`. 
The SAKURA autoencoder model will be built based on the parameters including `the number of encoder/decoder neurons, the dimension of latent space, the loss function, the regularization, and so on`.
To facilitate model training and testing, users also specify `optimizer settings, checkpoint and test segment, logging and result dumping settings`. 
All of these relevant settings can be conveniently specified in the model JSON configuration file, allowing them to be easily passed into SAKURA as a params dictionary instead of using an argument parser. 

<!--### Performing classification with trained netAE model
After training, one may want to use a classifier on the embedded space to test its classification accuracy. The `inference.py` script deals with comparing classification accuracy of netAE with other baseline models when using KNN and logistic regression, two simple classifiers. To start, make sure netAE, AE (the unsuperivsed counterpart), scVI, PCA, and ZIFA are trained and have their embedded spaces located in `MODEL_PATH`. Then simply pass in `--data-path`, `--model-path`, `--lab-size`, and `--dataset`. Additionally, to ensure that the labeled set used in training netAE is the same as here, make sure that you pass in the same seed `--seed` here as when training netAE.
-->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

