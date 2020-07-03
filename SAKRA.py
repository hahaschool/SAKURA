import argparse
import json
import random
import warnings

import numpy as np
import torch.backends.cudnn
import torch.cuda
import torch.optim
import torch.utils.data

from dataset import rna_count
from model_controllers.extractor_controller import ExtractorController
from models.extractor import Extractor
from utils.data_splitter import DataSplitter
from utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='model config JSON path')
    parser.add_argument('-v', '--verbose', type=bool, default=False, help='verbose console outputs')
    return parser.parse_args()

class SAKRA(object):
    def __init__(self, config_json_path, verbose=False):

        # Read configurations for arguments
        with open(config_json_path, 'r') as f:
            self.config = json.load(f)

        # Verbose (console) logging
        self.verbose = verbose

        # Logger working path
        self.log_path = self.config['log_path']

        # Device
        self.device = self.config['device']

        # Reproducibility
        if self.config['reproducible'] == 'True':
            self.rnd_seed = int(self.config['rnd_seed'])
            # When seed is set, turn on deterministic mode
            print("Reproducibe seed:", self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            random.seed(self.rnd_seed)
            if self.device == 'cuda':
                torch.cuda.manual_seed(self.rnd_seed)
                torch.cuda.manual_seed_all(self.rnd_seed)
                torch.backends.cudnn.enabled=False
                torch.backends.cudnn.benchmark=False
                torch.backends.cudnn.deterministic=True

        # Dataset
        if self.config['dataset']['type'] == 'rna_count':
            self.expr_csv_path = self.config['dataset']['expr_csv_path']
            self.pheno_csv_path = self.config['dataset']['pheno_csv_path']
            self.pheno_meta_path = self.config['dataset']['pheno_meta_path']
            self.signature_config_path = self.config['dataset']['signature_config_path']
            # TODO: load pre-defined splits (cell groups)
            # self.cell_group_config_path = self.config['dataset']['cell_group_config_path']
            # Import count data
            self.count_data = rna_count.SCRNASeqCountData(gene_csv_path=self.expr_csv_path,
                                                          pheno_csv_path=self.pheno_csv_path,
                                                          pheno_meta_json_path=self.pheno_meta_path,
                                                          mode='all',
                                                          verbose=self.verbose)

            # Selecting phenotypes and signatures to be used
            self.selected_pheno = self.config['dataset'].get('selected_pheno')
            if self.selected_pheno is None:
                self.selected_pheno = []
            self.selected_signature = self.config['dataset'].get('selected_signature')
            if self.selected_signature is None:
                self.selected_signature = []

            # Read gene signature metadata
            if len(self.signature_config_path) > 0:
                with open(self.signature_config_path, 'r') as f:
                    self.signature_config = json.load(f)

            # Subset phenotype and gene signature sets
            self.count_data.pheno_meta = {sel: self.count_data.pheno_meta[sel] for sel in self.selected_pheno}
            self.count_data.gene_meta = {sel: {
                'gene_list': self.signature_config[sel]['signature_list'],
                'pre_procedure': self.signature_config[sel]['pre_procedure'],
                'post_procedure': self.signature_config[sel]['post_procedure']
            } for sel in self.selected_signature}
            self.signature_config = {sel: self.signature_config[sel] for sel in self.selected_signature}

            # Build excluded_all gene set for signature supervision
            genes_to_exclude = list()
            for cur_signature in self.selected_signature:
                if self.signature_config[cur_signature]['exclude_from_input'] == 'True':
                    genes_to_exclude.extend(self.signature_config[cur_signature]['signature_list'])
            if len(genes_to_exclude) > 0:
                # Create excluded version of 'all' gene set
                self.count_data.gene_meta['all'] = {
                    'gene_list': '-',
                    'exclude_list': genes_to_exclude,
                    'pre_procedure': [],
                    'post_procedure': [{'type': 'ToTensor'}]
                }
            else:
                self.count_data.gene_meta['all'] = {
                    'gene_list': '*',
                    'pre_procedure': [],
                    'post_procedure': [{'type': 'ToTensor'}]
                }

            # Perform integrity check
            if self.integrity_check() is False:
                raise ValueError

        # Generate splits
        self.generate_splits()

        # Get actual gene count (input dimension)
        input_genes = self.count_data[0]['expr']['all'].shape[1]

        # Setup model
        self.model = Extractor(input_dim=input_genes,
                               signature_config=self.signature_config,
                               pheno_config=self.count_data.pheno_meta,
                               encoder_neurons=self.config['main_latent']['encoder_neurons'],
                               decoder_neurons=self.config['main_latent']['decoder_neurons'],
                               main_latent_dim=self.config['main_latent']['latent_dim'],
                               verbose=self.verbose)
        # Setup trainer
        self.controller = ExtractorController(model=self.model,
                                              config=self.config,
                                              pheno_config=self.count_data.pheno_meta,
                                              signature_config=self.signature_config,
                                              verbose=self.verbose)

        # Setup logger
        self.logger = Logger(log_path=self.log_path)

        # Save settings to log folder
        if self.config['dump_configs'] == 'True':
            self.logger.save_config(self.count_data.pheno_meta, self.log_path + '/pheno_config.json')
            self.logger.save_config(self.count_data.gene_meta, self.log_path + '/gene_meta.json')
            self.logger.save_config(self.signature_config, self.log_path + '/signature_config.json')
        if self.config['dump_splits'] == 'True':
            self.logger.save_splits(self.splits, self.log_path + '/splits.pkl')

        self.train_story(story=self.config['story'])


    def generate_splits(self):
        # Splits
        self.splits = dict()
        self.data_splitter = DataSplitter()

        ## Overall train/test split
        if self.config['overall_train_test_split']['type'] == 'auto':
            self.split_overall_train_dec = self.config['overall_train_test_split']['train_dec']
            self.split_overall_seed = self.config['overall_train_test_split']['seed']
            # Make overall train/test split
            all_mask = np.ones(len(self.count_data), dtype=np.int32)
            all_dec_bin = self.data_splitter.auto_random_k_bin_labelling(base=all_mask, k=10,
                                                                         seed=self.split_overall_seed)
            overall_train_test_split = self.data_splitter.get_incremental_train_test_split(base=all_dec_bin,
                                                                                           k=self.split_overall_train_dec)
            self.splits['overall_train'] = overall_train_test_split['train'].astype(np.bool)
            self.splits['overall_test'] = overall_train_test_split['test'].astype(np.bool)
        else:
            # TODO: Manual overall train/test split
            raise NotImplementedError

        ## Phenotype train/test
        if len(self.selected_pheno) > 0:
            for cur_pheno in self.selected_pheno:
                # Auto split
                if self.count_data.pheno_meta[cur_pheno]['split']['type'] == 'auto':
                    train_split_id = 'pheno_' + str(cur_pheno) + '_train'
                    test_split_id = 'pheno_' + str(cur_pheno) + '_test'
                    cur_base_split = self.splits[self.count_data.pheno_meta[cur_pheno]['split']['base']].astype(
                        np.int32)
                    cur_base_bin_marks = self.data_splitter.auto_random_k_bin_labelling(base=cur_base_split,
                                                                                        k=10,
                                                                                        seed=self.count_data.pheno_meta[
                                                                                            cur_pheno]['split']['seed'])
                    cur_label_train_test_split = self.data_splitter.get_incremental_train_test_split(
                        base=cur_base_bin_marks,
                        k=self.count_data.pheno_meta[cur_pheno]['split']['train_dec'])
                    self.splits[train_split_id] = cur_label_train_test_split['train'].astype(np.bool)
                    self.splits[test_split_id] = cur_label_train_test_split['test'].astype(np.bool)
                else:
                    # TODO: Manual label train/test split
                    raise NotImplementedError

        ## Signature train/test
        if len(self.selected_signature) > 0:
            for cur_signature in self.selected_signature:
                # Auto split
                if self.signature_config[cur_signature]['split']['type'] == 'auto':
                    train_split_id = 'signature_' + str(cur_signature) + '_train'
                    test_split_id = 'signature_' + str(cur_signature) + '_test'
                    cur_base_split = self.splits[self.signature_config[cur_signature]['split']['base']].astype(
                        np.int32)
                    cur_base_bin_marks = self.data_splitter.auto_random_k_bin_labelling(base=cur_base_split,
                                                                                        k=10,
                                                                                        seed=self.signature_config[
                                                                                            cur_signature]['split'][
                                                                                            'seed'])
                    cur_signature_train_test_split = self.data_splitter.get_incremental_train_test_split(
                        base=cur_base_bin_marks,
                        k=self.signature_config[cur_signature]['split']['train_dec'])
                    self.splits[train_split_id] = cur_signature_train_test_split['train'].astype(np.bool)
                    self.splits[test_split_id] = cur_signature_train_test_split['test'].astype(np.bool)
                else:
                    # TODO: Manual signature train/test split
                    raise NotImplementedError

        ## Print splits for debugging (in verbose mode)
        if self.verbose:
            print('==========================')
            print('Splits:')
            print(self.splits)

    def integrity_check(self):
        pheno_meta = self.count_data.pheno_meta
        gene_meta = self.count_data.gene_meta
        signature_config = self.signature_config

        ret = True

        # Phenotype checks
        # Check if all selected pheno exists
        problematic_phenos = list()
        for cur_pheno in self.selected_pheno:
            if (cur_pheno in pheno_meta.keys()) is False:
                problematic_phenos.append(cur_pheno)
        if len(problematic_phenos) > 0:
            warnings.warn("Exists selecting phenotype(s) not configured:" + str(problematic_phenos))
            ret = False

        # Check pheno_df_keys of selected phenotypes are exist in the dataset
        problematic_phenos = list()
        for cur_pheno in self.selected_pheno:
            cur_pheno_df_key = pheno_meta[cur_pheno]['pheno_df_key']
            if (cur_pheno_df_key in self.count_data.pheno_df.columns) is False:
                problematic_phenos.append(cur_pheno)
        if len(problematic_phenos) > 0:
            warnings.warn("Exists selecting phenotype(s) not found in the dataset:" + str(problematic_phenos))
            ret = False

        # Gene signature checks
        # Check if all selected gene signature sets exists in both configs
        problematic_signatures = list()
        for cur_signature in self.selected_signature:
            if (cur_signature not in gene_meta.keys()) or (cur_signature not in signature_config.keys()):
                problematic_signatures.append(cur_signature)
        if len(problematic_signatures) > 0:
            warnings.warn(
                "Exists signatures not consistent in gene_meta and signature_config:" + str(problematic_signatures))
            ret = False

        # Check consistency of gene signature sets
        problematic_signatures = list()
        for cur_signature in self.selected_signature:
            cur_gene_meta_contain = set(gene_meta[cur_signature]['gene_list'])
            cur_signature_config_contain = set(signature_config[cur_signature]['signature_list'])
            if cur_gene_meta_contain != cur_signature_config_contain:
                problematic_signatures.append(cur_signature)
        if len(problematic_signatures) > 0:
            warnings.warn("Genes in signature sets not consistent between gene_list and signature_config:" + str(
                problematic_signatures))
            ret = False

        # Check genes are all exist in the dataset
        problematic_signatures = list()
        for cur_signature in self.selected_signature:
            cur_gene_list = gene_meta[cur_signature]['gene_list']
            if all([x in self.count_data.gene_expr_mat.index for x in cur_gene_list]) is False:
                problematic_signatures.append(cur_signature)
        if len(problematic_signatures) > 0:
            warnings.warn("Genes in signature sets not exist in the dataset:" + str(problematic_signatures))
            ret = False

        if self.verbose:
            print('==========================')
            print("Configuration integrity pre-check ok:", ret)
        return ret

    def train(self,
              split_id,
              train_main=True,
              train_pheno=True, selected_pheno = None,
              train_signature=True, selected_signature=None,
              epoch=50, batch_size=100,
              tick_controller_epoch=True,
              make_logs=True, dump_latent=True, log_prefix='train', latent_prefix=''):
        """
        Batch train model for at least one epoch.
        :param split_id: (str) id of the split to be used in this train
        :param train_main: (bool) should forward the main latent space part
        :param train_pheno: (bool) should forward phenotype side tasks
        :param selected_pheno: (list of str, or str, or None) phenotype id(s) used for phenotype side task, selected phenotype(s)
         should be configured and in self.selected_pheno, if None, self.selected_pheno will take the place (i.e. train
         all phenotypes selected), this feature is designed for complex training where NN is partially forwarded
        :param train_signature: (bool) should forward gene signature side tasks
        :param selected_signature: (list of str, or str, or None) similar to selected_pheno, but for signature
        :param epoch: (int) epochs to be trained in this round of training
        :param batch_size: (int) batch size to be used in this round of training
        :param tick_controller_epoch: (bool) should controller epoch should be ticked (default: True)
        :param make_logs: (bool) should information, including losses should be logged (default: True)
        :param dump_latent: (bool) should all latent space representations be dumped after each batch (only cells within the split will be dumped)
        :param log_prefix: (str) the prefix of the training log (for losses, this prefix will be added first to the item name in tensorboard and filename of latent embeddings)
        :param latent_prefix: (str) the prefix to be added after log_prefix to latent embedding filename
        :return:
        """

        # Argument checks
        if train_pheno:
            if selected_pheno is None:
                selected_pheno = {idx: {'loss': '*', 'regularization': '*'} for idx in self.selected_pheno}
                warnings.warn(
                    "(To silence, specify phenotype selection explicitly in the config file.) Selecting all included phenotypes and linked losses and regularizations:" + str(
                        selected_pheno))
        else:
            if selected_pheno is not None:
                raise ValueError(
                    "Inconsistent training specification, specified phenotype to include in training but surpressed phenotype training.")

        if train_signature:
            if selected_signature is None:
                selected_signature = {idx: {'loss': '*', 'regularization': '*'} for idx in self.selected_signature}
                warnings.warn(
                    "(To silence, specify signature selection explicitly in the config file.) Selecting all included signatures and linked losses and regularizations: " + str(
                        selected_signature))
        else:
            if selected_signature is not None:
                raise ValueError(
                    "Inconsistent training specification, specified signature to include in training but surpressed signature training.")
        if batch_size is None:
            warnings.warn(
                "(To slience, specify batch_size explicitly in the config file.) Using default batch_size 50.")
            batch_size = 50

        # Split mask
        selected_split_mask = self.splits[split_id]

        # Train epochs
        for cur_epoch in range(epoch):
            # Begin epoch
            if tick_controller_epoch:
                self.controller.next_epoch(prog_main=train_main,
                                           prog_pheno=train_pheno, selected_pheno=selected_pheno,
                                           prog_signature=train_signature, selected_signature=selected_signature)
            # TODO: selective training
            # Set sampler
            sampler = torch.utils.data.BatchSampler(
                sampler=torch.utils.data.SubsetRandomSampler(
                    np.arange(len(self.count_data))[selected_split_mask]
                ),
                batch_size=batch_size,
                drop_last=False
            )

            for cur_idx in sampler:
                cur_batch = self.count_data[cur_idx]
                controller_ret = self.controller.train(batch=cur_batch,
                                                       backward_reconstruction_loss=train_main,
                                                       backward_main_latent_regularization=train_main,
                                                       backward_pheno_loss=train_pheno, selected_pheno=selected_pheno,
                                                       backward_signature_loss=train_signature,
                                                       selected_signature=selected_signature)
                if make_logs:
                    self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                         loss_name_prefix=log_prefix)
                self.controller.tick()

            if dump_latent:
                # Forward the whole split to the model
                controller_ret = self.controller.eval(self.count_data[selected_split_mask],
                                                      forward_pheno=True, selected_pheno=None,
                                                      forward_signature=True, selected_signature=None,
                                                      forward_reconstruction=True, forward_main_latent=True,
                                                      dump_latent=True)
                # TODO: use multithreading to split dumping from the main thread (maybe implement into Logger, rather than here)
                self.logger.dump_latent_to_csv(controller_output=controller_ret,
                                               dump_main=True,
                                               dump_pheno=True, selected_pheno=self.selected_pheno,
                                               dump_signature=True, selected_signature=self.selected_signature,
                                               rownames=self.count_data.gene_expr_mat.columns[selected_split_mask],
                                               path=self.log_path + '/' + log_prefix + latent_prefix + str(
                                                   cur_epoch) + '_latent.csv')

    def test(self, split_id,
             test_main=True,
             test_pheno=True, selected_pheno=None,
             test_signature=True, selected_signature=None,
             make_logs=False, dump_latent=True, log_prefix='test', latent_prefix=''):
        # Split mask
        selected_split_mask = self.splits[split_id]

        # Eval on split
        controller_ret = self.controller.eval(self.count_data[selected_split_mask],
                                              forward_signature=test_signature,
                                              selected_signature=selected_signature,
                                              forward_pheno=test_pheno, selected_pheno=selected_pheno,
                                              forward_reconstruction=test_main, forward_main_latent=True,
                                              dump_latent=dump_latent)

        # Log losses in tensorboard (by using Logger)
        if make_logs:
            self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                 loss_name_prefix=log_prefix)
            self.controller.tick()

        # Dump latent space into CSV files
        if dump_latent:
            self.logger.dump_latent_to_csv(controller_output=controller_ret,
                                           dump_main=True,
                                           dump_pheno=True, selected_pheno=self.selected_pheno,
                                           dump_signature=True, selected_signature=self.selected_signature,
                                           rownames=self.count_data.gene_expr_mat.columns[selected_split_mask],
                                           path=self.log_path+'/'+log_prefix+latent_prefix+'_latent.csv')



    def train_story(self, story:list):
        for cur_story_item in story:
            # Verbose logging
            if self.verbose:
                print("Training story:", cur_story_item.get('remark'))
                print(cur_story_item)

            # Select specified split
            self.train(split_id=cur_story_item['use_split'],
                       train_main=(cur_story_item['train_main_latent'] == 'True'),
                       train_pheno=(cur_story_item['train_pheno'] == 'True'),
                       train_signature=(cur_story_item['train_signature'] == 'True'),
                       selected_pheno=cur_story_item.get('selected_pheno'),
                       selected_signature=cur_story_item.get('selected_signature'),
                       epoch=cur_story_item['epochs'],
                       batch_size=cur_story_item.get('batch_size'))


if __name__ == '__main__':
    print('SCARE/SAKRA Prototype')
    print('Loading dataset...')
    args = parse_args()
    instance = SAKRA(config_json_path=args.config, verbose=args.verbose)
