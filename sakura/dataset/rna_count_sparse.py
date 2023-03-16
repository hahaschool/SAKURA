import json

import numpy
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
from torch.utils.data import Dataset

from sakura.utils.data_transformations import ToBinary
from sakura.utils.data_transformations import ToKBins
# Transformations
from sakura.utils.data_transformations import ToOnehot
from sakura.utils.data_transformations import ToOrdinal
from sakura.utils.data_transformations import ToTensor


class SCRNASeqCountDataSparse(Dataset):
    "Accepts matrixMM (could be dgcmatrix in R) as data contained (will still load everything into memory, but using sparse matrix now)."

    def __init__(self, gene_MM_path, gene_name_csv_path, cell_name_csv_path,
                 pheno_csv_path, pheno_df_dtype=None, pheno_df_na_filter=True,
                 gene_meta_json_path=None, pheno_meta_json_path=None,
                 gene_meta=None, pheno_meta=None,
                 mode='all', verbose=False):

        # Verbose console logging
        self.verbose = verbose

        # Persist argument list
        self.gene_MM_path = gene_MM_path
        self.gene_name_csv_path = gene_name_csv_path
        self.cell_name_csv_path = cell_name_csv_path
        self.gene_meta_json_path = gene_meta_json_path
        self.pheno_csv_path = pheno_csv_path
        self.pheno_meta_json_path = pheno_meta_json_path
        self.mode = mode

        # Register transformers
        self.to_tensor = ToTensor()
        self.to_onehot = ToOnehot()
        self.to_ordinal = ToOrdinal()
        self.to_kbins = ToKBins()
        self.to_binary = ToBinary()

        # Read gene expression matrix (from MM .mtx file, as sparse matrix)
        if verbose:
            print("Loading expression matrix from mtx (MM) file.")
        # Read sparse matrix and convert to csc_matrix
        # csc_matrix is required by pandas, also efficient when cells are stored in column side
        self._gene_expr_mat_orig = scipy.io.mmread(self.gene_MM_path)
        # Read gene names
        self.gene_names = pd.read_csv(self.gene_name_csv_path, index_col=0)
        # Read cell names
        self.cell_names = pd.read_csv(self.cell_name_csv_path, index_col=0)
        # Build pandas sparse DataFrame
        if isinstance(self._gene_expr_mat_orig, np.ndarray):
            # Fallback to dense matrix
            self._gene_expr_mat_orig = pd.DataFrame(data=self._gene_expr_mat_orig,
                                                    index=self.gene_names.iloc[:, 0],
                                                    columns=self.cell_names.iloc[:, 0])
        else:
            self._gene_expr_mat_orig = pd.DataFrame.sparse.from_spmatrix(data=self._gene_expr_mat_orig,
                                                                         index=self.gene_names.iloc[:, 0],
                                                                         columns=self.cell_names.iloc[:, 0])

        self._gene_expr_mat_orig.columns.name = 'cell'
        self._gene_expr_mat_orig.index.name = 'gene'
        self.gene_expr_mat = self._gene_expr_mat_orig.copy()

        # Note that sparse dataframe is doing row slicing very slowly, so for each expression set,
        # should pre-slice the whole sparse dataframe

        if self.verbose:
            print('==========================')
            print('rna_count dataset (sparse MM version):')
            print("Imported gene expression matrix from:", self.gene_MM_path)
            print("Imported gene names from:", self.gene_name_csv_path)
            print("Imported cell names from:", self.cell_name_csv_path)
            print(self.gene_expr_mat.shape)
            print(self.gene_expr_mat.iloc[:, 0:3])

        # Read gene expression matrix metadata
        self.gene_meta = gene_meta
        self.flag_expr_set_pre_sliced = False
        self.expr_mat_pre_sliced = dict()
        if self.gene_meta is None:
            self.gene_meta = {
                "all": {
                    "gene_list": "*",
                    "pre_procedure": [],
                    'post_procedure': [
                        {
                            "type": "ToTensor"
                        }
                    ]
                }
            }
            if self.verbose:
                print('No external gene expression set provided, using dummy.')
        if self.gene_meta_json_path is not None:
            with open(self.gene_meta_json_path, 'r') as f:
                self.gene_meta = json.load(f)
                if self.verbose:
                    print("Gene expression set metadata imported from:", self.gene_meta_json_path)
        if self.verbose:
            print("Gene expression set metadata:")
            print(self.gene_meta)

        # Read phenotype data frame
        self._pheno_df_orig = pd.read_csv(self.pheno_csv_path, index_col=0, header=0, dtype=pheno_df_dtype, na_filter=pheno_df_na_filter)
        self.pheno_df = self._pheno_df_orig.copy()

        if self.verbose:
            print("Phenotype data from CSV from:", self.pheno_csv_path)
            print(self.pheno_df.shape)
            print(self.pheno_df)

        # Read phenotype colmun metadata
        self.pheno_meta = pheno_meta
        if self.pheno_meta is None:
            self.pheno_meta = {}
        if pheno_meta_json_path is not None:
            if self.verbose:
                print("Reading phenotype metadata json from:", self.pheno_meta_json_path)
            with open(self.pheno_meta_json_path, 'r') as f:
                self.pheno_meta = json.load(f)

        # Cell list
        self._cell_list_orig = self._gene_expr_mat_orig.columns.values
        self.cell_list = self._cell_list_orig.copy()

        # Sanity check
        # Cell should be consistent between expr matrix and phenotype table
        if (self._gene_expr_mat_orig.columns.values != self._pheno_df_orig.index.values).any():
            raise ValueError

    def expr_set_pre_slice(self):
        self.flag_expr_set_pre_sliced = True
        self.expr_mat_pre_sliced = dict()
        for cur_expr_key in self.gene_meta.keys():
            cur_expr_meta = self.gene_meta[cur_expr_key]
            if type(cur_expr_meta['gene_list']) is list or type(cur_expr_meta['gene_list']) is np.ndarray:
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.loc[cur_expr_meta['gene_list'], :].copy()
            elif cur_expr_meta['gene_list'] == '*':
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.copy()
            elif cur_expr_meta['gene_list'] == '-':
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.drop(cur_expr_meta['exclude_list'],
                                                                                 axis=0).copy()

    def __select_expr_mat(self, cur_expr_key, item):
        cur_expr_mat = None
        if self.flag_expr_set_pre_sliced:
            # Select pre-sliced expression matrices
            cur_expr_mat = self.expr_mat_pre_sliced[cur_expr_key].iloc[:, item].copy()
        else:
            # Ad-lib slice expression matrix
            cur_expr_meta = self.gene_meta[cur_expr_key]
            # Gene Selection
            if type(cur_expr_meta['gene_list']) is list or type(cur_expr_meta['gene_list']) is np.ndarray:
                # Select given genes
                cur_expr_mat = self.gene_expr_mat.loc[cur_expr_meta['gene_list'], :].iloc[:, item].copy()
            elif cur_expr_meta['gene_list'] == '*':
                # Select all genes
                cur_expr_mat = self.gene_expr_mat.iloc[:, item].copy()
            elif cur_expr_meta['gene_list'] == '-':
                # Deselect given genes
                cur_expr_mat = self.gene_expr_mat.drop(cur_expr_meta['exclude_list'], axis=0).iloc[:, item].copy()
        return cur_expr_mat

    def export_data(self, item,
                    include_raw=True,
                    include_proc=True,
                    include_cell_key=True,
                    include_item=True):
        """
        Export a batch of data given 'item' as index.
        :param item: index
        :param include_raw: should the unprocessed, subsetted expression matrix and phenotype data frame be exported
        :param include_proc: should the processed data (following procedures specified in the configs) be exported
        :param include_cell_key: should names/keys of the cells be exported
        :return:
        """

        # Type adaptation: when 'item' is a single index, convert it to list
        if type(item) is int:
            item = [item]

        # Extract required cells and prepare required structure
        ret = dict()

        if include_cell_key is True:
            ret['cell_key'] = self.gene_expr_mat.columns.values[item]

        if include_item is True:
            ret['item'] = item

        # Prepare raw gene output (if needed)
        if include_raw is True:
            ret['expr_mat'] = self.gene_expr_mat.iloc[:, item].copy()
        if include_proc is True:
            ret['expr'] = dict()
            for cur_expr_key in self.gene_meta.keys():
                cur_expr_meta = self.gene_meta[cur_expr_key]
                ret['expr'][cur_expr_key] = self.__select_expr_mat(cur_expr_key, item)
                # Post Transformation
                for cur_procedure in cur_expr_meta['post_procedure']:
                    if cur_procedure['type'] == 'ToBinary':
                        ret['expr'][cur_expr_key] = self.to_binary(ret['expr'][cur_expr_key],
                                                                   threshold=cur_procedure.get('threshold'),
                                                                   inverse=(cur_procedure.get('inverse') == 'True'),
                                                                   scale_factor=cur_procedure.get('scale_factor', 1.0))
                    elif cur_procedure['type'] == 'ToTensor':
                        ret['expr'][cur_expr_key] = self.to_tensor(ret['expr'][cur_expr_key],
                                                                   input_type='gene',
                                                                   force_tensor_type=cur_procedure.get('force_tensor_type'))
                    else:
                        print("Unsupported post-transformation")
                        raise NotImplementedError

        # Prepare phenotype output
        if include_raw is True:
            ret['pheno_df'] = self.pheno_df.iloc[item, :].copy()
        if include_proc is True:
            ret['pheno'] = dict()
            for pheno_output_key in self.pheno_meta.keys():
                cur_pheno_meta = self.pheno_meta[pheno_output_key]

                # Slice phenotype dataframe
                if cur_pheno_meta['type'] == 'categorical':
                    ret['pheno'][pheno_output_key] = self.pheno_df.loc[:, [cur_pheno_meta['pheno_df_key']]].iloc[item, :].copy()
                    # Process phenotype label as required
                    for cur_procedure in cur_pheno_meta['post_procedure']:
                        if cur_procedure['type'] == 'ToTensor':
                            ret['pheno'][pheno_output_key] = self.to_tensor(sample=ret['pheno'][pheno_output_key],
                                                                            input_type='pheno',
                                                                            force_tensor_type=cur_procedure.get('force_tensor_type'))
                        elif cur_procedure['type'] == 'ToOnehot':
                            ret['pheno'][pheno_output_key] = self.to_onehot(sample=ret['pheno'][pheno_output_key],
                                                                            order=cur_pheno_meta['order'])
                        elif cur_procedure['type'] == 'ToOrdinal':
                            ret['pheno'][pheno_output_key] = self.to_ordinal(sample=ret['pheno'][pheno_output_key],
                                                                             order=cur_pheno_meta['order'])
                        elif cur_procedure['type'] == 'ToKBins':
                            ret['pheno'][pheno_output_key] = self.to_kbins(sample=ret['pheno'][pheno_output_key],
                                                                           n_bins=cur_procedure['n_bins'],
                                                                           encode=cur_procedure['encode'],
                                                                           strategy=cur_procedure['strategy'])
                        elif cur_procedure['type'] == 'ToBinary':
                            ret['pheno'][pheno_output_key] = self.to_binary(sample=ret['pheno'][pheno_output_key],
                                                                            threshold=cur_procedure.get('threshold'),
                                                                            inverse=(cur_procedure.get('inverse') == 'True'),
                                                                            scale_factor=cur_procedure.get('scale_factor', 1.0))
                        else:
                            raise NotImplementedError('Unsupported transformation type for phenotype groups')
                elif cur_pheno_meta['type'] == 'numerical':
                    ret['pheno'][pheno_output_key] = self.pheno_df.loc[:, cur_pheno_meta['pheno_df_keys']].iloc[item, :].copy()
                    # Process phenotype label as required
                    for cur_procedure in cur_pheno_meta['post_procedure']:
                        if cur_procedure['type'] == 'ToTensor':
                            ret['pheno'][pheno_output_key] = self.to_tensor(sample=ret['pheno'][pheno_output_key],
                                                                            input_type='pheno',
                                                                            force_tensor_type=cur_procedure.get('force_tensor_type'))
                        elif cur_procedure['type'] == 'ToBinary':
                            ret['pheno'][pheno_output_key] = self.to_binary(sample=ret['pheno'][pheno_output_key],
                                                                            threshold=cur_procedure.get('threshold'),
                                                                            inverse=(cur_procedure.get('inverse') == 'True'),
                                                                            scale_factor=cur_procedure.get('scale_factor', 1.0))
                        else:
                            raise NotImplementedError('Unsupported transformation type for phenotype groups')
                else:
                    raise ValueError('Unsupported phenotype group name')

        return ret

    def collate_fn(self, batch):
        # Assemble individual cell keys and export a whole set
        collated_item = list()
        for sample in batch:
            sample['item'] = numpy.array(sample['item']).squeeze()
            collated_item.append(sample['item'])
        return self.export_data(collated_item,
                                include_raw=True,
                                include_proc=True,
                                include_cell_key=True)

    def __len__(self):
        # Length of the dataset is considered as the number of cells
        return self.gene_expr_mat.shape[1]

    def __getitem__(self, item):
        if self.mode == 'all':
            return self.export_data(item,
                                    include_raw=True,
                                    include_proc=True,
                                    include_cell_key=True)
        elif self.mode == 'key':
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=False,
                                    include_cell_key=True)
        elif self.mode == 'index':
            return
        else:
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=True,
                                    include_cell_key=False)
