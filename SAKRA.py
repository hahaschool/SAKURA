import argparse
import json
import os
import pickle
import pprint
import random
import warnings
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.optim
import torch.utils.data
import torch.multiprocessing
from copy import deepcopy

from dataset import rna_count
from dataset import rna_count_sparse
from model_controllers.extractor_controller import ExtractorController
from models.extractor import Extractor
from utils.data_splitter import DataSplitter
from utils.logger import Logger

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='model config JSON path')
    parser.add_argument('-v', '--verbose', type=bool, default=False, help='verbose console outputs')
    parser.add_argument('-s', '--suppress_train', type=bool, default=False, help='suppress model training, only setup dataset and model')
    parser.add_argument('-r', '--resume', type=str, default='', help='resume training process from saved checkpoint file')
    parser.add_argument('-i', '--inference', type=str, default='', help='perform inference from saved checkpoint file containing models')
    parser.add_argument('-y', '--inference_story', type=str, default='./inference.json', help='story file of inference')
    parser.add_argument('-x', '--suppress_tensorboardX', type=bool, default=False, help='suppress Logger to initiate tensorboardX (to prevent flushing logs)')
    parser.add_argument('-e', '--external_module', type=bool, default=False, help='insert modules from external (pretrained) models')
    parser.add_argument('-E', '--external_module_path', type=str, default='./insert_config.json', help='path of external model config')
    return parser.parse_args()


class SAKRA(object):
    def __init__(self, config_json_path, verbose=False,
                 suppress_train=False, suppress_tensorboardX=False):

        # Read configurations for arguments
        with open(config_json_path, 'r') as f:
            self.config = json.load(f)

        # Verbose (console) logging
        self.verbose = verbose

        # Logger working path
        self.log_path = self.config['log_path']

        # Device
        self.device = self.config['device']

        # Persistant test set optimization
        self.persist_test_set = (self.config.get('persist_test_set') == 'True')
        self.persisted_batches = dict()

        # Reproducibility
        if self.config['reproducible'] == 'True':
            self.rnd_seed = int(self.config['rnd_seed'])
            # When seed is set, turn on deterministic mode
            logger.info("Reproducibe seed: {}", self.rnd_seed)

            torch.manual_seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            random.seed(self.rnd_seed)
            os.environ['PYTHONHASHSEED'] = str(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # torch.use_deterministic_algorithms(True)

        # Setup dataset
        self.setup_dataset()

        # Generate splits
        self.generate_splits()

        # Get actual gene count (input dimension)
        input_genes = self.count_data[0]['expr']['all'].shape[1]
        if self.verbose:
            logger.debug('Input gene number for building model: {}', input_genes)

        # Setup model
        self.model = Extractor(input_dim=input_genes,
                               signature_config=self.signature_config,
                               pheno_config=self.count_data.pheno_meta,
                               main_lat_config=self.config['main_latent'],
                               pre_encoder_config=self.config.get('pre_encoder_config'),
                               verbose=self.verbose)
        # Setup trainer
        self.controller = ExtractorController(model=self.model,
                                              config=self.config,
                                              pheno_config=self.count_data.pheno_meta,
                                              signature_config=self.signature_config,
                                              verbose=self.verbose)

        # Setup logger
        self.logger = Logger(log_path=self.log_path, suppress_tensorboardX=suppress_tensorboardX)

        # Save settings to log folder
        if self.config['dump_configs'] == 'True':
            self.logger.save_config(self.count_data.pheno_meta, self.log_path + '/pheno_config.json')
            self.logger.save_config(self.count_data.gene_meta, self.log_path + '/gene_meta.json')
            self.logger.save_config(self.signature_config, self.log_path + '/signature_config.json')
        if self.config['dump_splits'] == 'True':
            self.logger.save_splits(self.splits, self.log_path + '/splits.pkl')

        # suppress training if needed (debug, or in case to resume)
        if suppress_train:
            return

        self.train_story(story=self.config['story'])

    def setup_dataset(self):
        # Dataset (main part)
        if self.config['dataset']['type'] == 'rna_count':
            self.expr_csv_path = self.config['dataset'].get('expr_csv_path')
            self.pheno_csv_path = self.config['dataset'].get('pheno_csv_path')
            self.pheno_meta_path = self.config['dataset'].get('pheno_meta_path')
            self.signature_config_path = self.config['dataset'].get('signature_config_path')
            self.pheno_df_dtype = self.config['dataset'].get('pheno_df_dtype')
            self.pheno_df_na_filter = self.config['dataset'].get('pheno_df_na_filter') == 'True'
            # TODO: load pre-defined splits (cell groups)
            # self.cell_group_config_path = self.config['dataset']['cell_group_config_path']
            # Import count data
            self.count_data = rna_count.SCRNASeqCountData(gene_csv_path=self.expr_csv_path,
                                                          pheno_csv_path=self.pheno_csv_path,
                                                          pheno_df_dtype=self.pheno_df_dtype,
                                                          pheno_meta_json_path=self.pheno_meta_path,
                                                          pheno_df_na_filter=self.pheno_df_na_filter,
                                                          mode='all',
                                                          verbose=self.verbose)
        elif self.config['dataset']['type'] == 'rna_count_sparse':
            # Persist arguments
            self.gene_expr_MM_path = self.config['dataset'].get('gene_expr_MM_path')
            self.gene_name_csv_path = self.config['dataset'].get('gene_name_csv_path')
            self.cell_name_csv_path = self.config['dataset'].get('cell_name_csv_path')
            self.pheno_csv_path = self.config['dataset'].get('pheno_csv_path')
            self.pheno_meta_path = self.config['dataset'].get('pheno_meta_path')
            self.signature_config_path = self.config['dataset'].get('signature_config_path')
            self.pheno_df_dtype = self.config['dataset'].get('pheno_df_dtype')
            self.pheno_df_na_filter = self.config['dataset'].get('pheno_df_na_filter') == 'True'

            self.count_data = rna_count_sparse.SCRNASeqCountDataSparse(gene_MM_path=self.gene_expr_MM_path,
                                                                       gene_name_csv_path=self.gene_name_csv_path,
                                                                       cell_name_csv_path=self.cell_name_csv_path,
                                                                       pheno_csv_path=self.pheno_csv_path,
                                                                       pheno_df_dtype=self.pheno_df_dtype,
                                                                       pheno_df_na_filter=self.pheno_df_na_filter,
                                                                       pheno_meta_json_path=self.pheno_meta_path,
                                                                       mode='all',
                                                                       verbose=self.verbose)
        else:
            raise ValueError('Unrecognized dataset type')

        # Dataset (Phenotype and Signature)
        if self.config['dataset']['type'] == 'rna_count' or self.config['dataset']['type'] == 'rna_count_sparse':
            # Selecting phenotypes and signatures to be used
            self.selected_pheno = self.config['dataset'].get('selected_pheno')
            if self.selected_pheno is None:
                self.selected_pheno = []
            self.selected_signature = self.config['dataset'].get('selected_signature')
            if self.selected_signature is None:
                self.selected_signature = []

            # Read gene signature metadata
            if self.signature_config_path is not None and len(self.signature_config_path) > 0:
                with open(self.signature_config_path, 'r') as f:
                    self.signature_config = json.load(f)
            else:
                self.signature_config = {}

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

            # For sparse data, perform pre-slice to optimize performance
            if self.config['dataset']['type'] == 'rna_count_sparse':
                if self.config['dataset']['expr_mat_pre_slice'] == 'True':
                    self.count_data.expr_set_pre_slice()

            # Perform integrity check
            if self.integrity_check() is False:
                raise ValueError('Integrity check failed, see console log for details')
        else:
            raise ValueError('Unsupported dataset type')

    def generate_splits(self):
        # Splits
        self.splits = dict()
        self.data_splitter = DataSplitter()

        # 'all' split mask for selecting all data
        self.splits['all'] = np.ones(len(self.count_data), dtype=np.bool)

        if self.config.get('manual_split') == 'True':
            # Should load external files for splits
            # when conflicts occur, will directly override existing splits (i.e. all)
            with open(self.config.get('manual_split_pkl_path'), 'rb') as f:
                ext_splits = pickle.load(f)
            for cur_split_key in ext_splits.keys():
                self.splits[cur_split_key] = ext_splits[cur_split_key]
            if self.verbose:
                logger.debug("Imported external splits from {}", self.config.get('manual_split_pkl_path'))
                logger.debug("External splits: {}", ext_splits)
                logger.debug("Merged splits: {}", self.splits)

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
        elif self.config['overall_train_test_split']['type'] == 'none':
            # Do nothing
            if self.verbose:
                logger.debug('Skipped main splits')
        else:
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
                elif self.count_data.pheno_meta[cur_pheno]['split']['type'] == 'none':
                    # Do nothing
                    if self.verbose:
                        logger.debug('Skipped pheno splits for: {}', cur_pheno)
                else:
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
                elif self.signature_config[cur_signature]['split']['type'] == 'none':
                    # Do nothing
                    if self.verbose:
                        logger.debug('Skipped signature splits for: {}', cur_signature)
                else:
                    raise NotImplementedError

        ## Print splits for debugging (in verbose mode)
        if self.verbose:
            logger.debug('Splits: {}', self.splits)

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
            if pheno_meta[cur_pheno]['type'] == 'categorical':
                cur_pheno_df_key = pheno_meta[cur_pheno]['pheno_df_key']
                if (cur_pheno_df_key in self.count_data.pheno_df.columns) is False:
                    problematic_phenos.append(cur_pheno)
            elif pheno_meta[cur_pheno]['type'] == 'numerical':
                cur_pheno_df_keys = pheno_meta[cur_pheno]['pheno_df_keys']
                for cur_key in cur_pheno_df_keys:
                    if (cur_key in self.count_data.pheno_df.columns) is False:
                        problematic_phenos.append(cur_pheno)
            else:
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

        logger.info("Configuration integrity pre-check: {}", ret)
        return ret

    def train(self,
              split_id,
              train_main=True,
              train_pheno=True, selected_pheno=None,
              train_signature=True, selected_signature=None,
              epoch=50, batch_size=100,
              tick_controller_epoch=True,
              make_logs=True, log_prefix='train', log_loss_groups=['loss', 'regularization'], save_raw_loss=False,
              dump_latent=True, latent_prefix='',
              test_every_epoch=False, test_on_segment=False, test_segment=2000, tests=None,
              checkpoint_on_segment=False, checkpoint_segment=2000, checkpoint_prefix='', checkpoint_save_arch=False,
              resume=False, resume_dict=None,
              detach=False, detach_from=''):
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
        :param test_every_epoch: (bool) should test/evaluation be performed after finishing each epochs
        :param tests: (list of dict) test configurations
        :param checkpoint_on_segment: (bool) should model be checkpointed after a certain tick interval
        :param checkpoint_segment: (int) checkpoint segment length
        :param checkpoint_prefix: (int) filename prefix for checkpoint files
        :param resume: (bool) should resume from saved training session
        :param resume_dict: (bool) session state dict used for resuming previous training
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

        # Setup sampler
        sampler = torch.utils.data.BatchSampler(
            sampler=torch.utils.data.SubsetRandomSampler(
                np.arange(len(self.count_data))[selected_split_mask]
            ),
            batch_size=batch_size,
            drop_last=False
        )

        # Local tick for segmental test
        cur_tick = 0
        cur_epoch = 0

        # Resume flag
        to_resume_flag = resume

        # Handle resume
        if resume and to_resume_flag:
            cur_epoch = resume_dict['cur_epoch']

        # Train epochs
        while cur_epoch < epoch:
            # Persist training state for resume
            resume_dict['cur_epoch'] = cur_epoch

            # Handle resuming/persisting sampler
            if resume and to_resume_flag:
                sampler = resume_dict['sampler']
                idx_list = resume_dict['idx_list']
                # When resume, start from the next tick/idx_idx
                cur_tick = resume_dict['cur_tick']
                idx_idx = resume_dict['idx_idx']
                # Resume finished
                to_resume_flag = False
            else:
                # Preload indices for resumption
                idx_list = list()
                for cur_idx in iter(sampler):
                    idx_list.append(cur_idx)

                # Persist sampler and index for resume
                resume_dict['idx_list'] = idx_list
                resume_dict['sampler'] = sampler
                idx_idx = 0

            for cur_idx in idx_list[idx_idx:]:
                cur_batch = self.count_data[cur_idx]
                controller_ret = self.controller.train(batch=cur_batch,
                                                       backward_reconstruction_loss=train_main,
                                                       backward_main_latent_regularization=train_main,
                                                       backward_pheno_loss=train_pheno, selected_pheno=selected_pheno,
                                                       backward_signature_loss=train_signature,
                                                       selected_signature=selected_signature,
                                                       detach=detach, detach_from=detach_from, save_raw_loss=save_raw_loss)

                # Verbose logging
                if self.verbose:
                    logger.debug("cur_batch['cell_key'][:10]: {}", cur_batch['cell_key'][:10])

                # Make logs
                if make_logs:
                    self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                         loss_name_prefix=log_prefix, selected_loss_group=log_loss_groups)

                # Segmental Test
                if test_on_segment and (cur_tick + 1) % test_segment == 0:
                    # Perform test
                    for cur_test in tests:
                        # When test, all latents will be evaluated
                        self.test(split_id=cur_test['on_split'],
                                  test_main=True,
                                  test_pheno=True, selected_pheno=None,
                                  test_signature=True, selected_signature=None,
                                  make_logs=(cur_test.get('make_logs') == 'True'),
                                  log_prefix=cur_test.get('log_prefix', 'test'),
                                  log_loss_groups=log_loss_groups,
                                  dump_latent=(cur_test.get('dump_latent') == 'True'),
                                  latent_prefix=cur_test.get('latent_prefix', ''),
                                  save_raw_loss=save_raw_loss)

                # Tick controller
                self.controller.tick()

                # When checkpoint: current epoch, current tick, current idx_idx, model ticked
                # When resume: stay epoch, next tick, next idx_idx, stay model (as it has already been ticked)
                cur_tick += 1
                idx_idx += 1

                # Persist indices for resuming
                resume_dict['cur_tick'] = cur_tick
                resume_dict['idx_idx'] = idx_idx

                # Segmental checkpoint
                if checkpoint_on_segment and cur_tick % checkpoint_segment == 0:
                    # Save checkpoint
                    self.save_checkpoint(training_state=resume_dict,
                                         checkpoint_path=self.log_path + checkpoint_prefix + '_tick_' + str(cur_tick) + '.pth',
                                         save_model_arch=checkpoint_save_arch)

            # Begin new epoch (if resuming, skip)
            if tick_controller_epoch:
                self.controller.next_epoch(prog_main=train_main,
                                           prog_pheno=train_pheno, selected_pheno=selected_pheno,
                                           prog_signature=train_signature, selected_signature=selected_signature)
            cur_epoch += 1
            resume_dict['cur_epoch'] = cur_epoch

            # Epoch-wise test
            if test_every_epoch:
                for cur_test in tests:
                    # When test, all latents will be evaluated
                    self.test(split_id=cur_test['on_split'],
                              test_main=True,
                              test_pheno=True, selected_pheno=None,
                              test_signature=True, selected_signature=None,
                              make_logs=(cur_test.get('make_logs') == 'True'),
                              log_prefix=cur_test.get('log_prefix', 'test'),
                              log_loss_groups=log_loss_groups,
                              dump_latent=(cur_test.get('dump_latent') == 'True'),
                              latent_prefix=cur_test.get('latent_prefix', ''),
                              save_raw_loss=save_raw_loss)

        # Terminal test if segmental test is on
        if test_on_segment and (test_every_epoch == False):
            # Perform test
            for cur_test in tests:
                # When test, all latents will be evaluated
                self.test(split_id=cur_test['on_split'],
                          test_main=True,
                          test_pheno=True, selected_pheno=None,
                          test_signature=True, selected_signature=None,
                          make_logs=(cur_test.get('make_logs') == 'True'),
                          log_prefix=cur_test.get('log_prefix', 'test'),
                          log_loss_groups=log_loss_groups,
                          dump_latent=(cur_test.get('dump_latent') == 'True'),
                          latent_prefix=cur_test.get('latent_prefix', ''),
                          save_raw_loss=save_raw_loss)

        # Terminal checkpoint when segmental checkpoint is on
        if checkpoint_on_segment:
            # Save checkpoint
            self.save_checkpoint(training_state=resume_dict,
                                 checkpoint_path=self.log_path + checkpoint_prefix + '_tick_' + str(cur_tick) + '.pth',
                                 save_model_arch=checkpoint_save_arch)

    def test(self, split_id,
             test_main=True,
             test_pheno=True, selected_pheno=None,
             test_signature=True, selected_signature=None,
             make_logs=False, log_prefix='test', log_loss_groups=['loss', 'regularization'],
             dump_latent=True, latent_prefix='',
             dump_pre_encoder_output=False,
             dump_reconstructed_output=False, reconstructed_output_naming='dimid',
             dump_predicted_phenos=False,
             dump_predicted_signatures=False,
             compression='none', save_raw_loss=False):

        selected_split_mask = self.splits[split_id]

        # Persistant test set optimization
        cur_batch = self.persisted_batches.get(split_id)

        if cur_batch is None:
            # Split mask
            cur_batch = self.count_data[selected_split_mask]
            if self.persist_test_set:
                self.persisted_batches[split_id] = cur_batch

        # Eval on split
        controller_ret = self.controller.eval(cur_batch,
                                              forward_signature=test_signature, selected_signature=selected_signature,
                                              forward_pheno=test_pheno, selected_pheno=selected_pheno,
                                              forward_reconstruction=test_main, forward_main_latent=True,
                                              dump_latent=dump_latent, save_raw_loss=save_raw_loss)

        # Log losses in tensorboard (by using Logger)
        if make_logs:
            self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                 loss_name_prefix=log_prefix, selected_loss_group=log_loss_groups)
            self.controller.tick()

        # Handle CSV path and filename (in case of compression)
        if compression == 'hdf':
            csv_path = self.log_path + '/' + str(self.controller.cur_epoch) + '_' + latent_prefix + '.h5'
        elif compression == 'gzip':
            csv_path = self.log_path + '/' + str(self.controller.cur_epoch) + '_' + latent_prefix + '.csv.gz'
        else:
            csv_path = self.log_path + '/' + str(self.controller.cur_epoch) + '_' + latent_prefix + '.csv'

        # Dump latent space into CSV files
        if dump_latent or \
                dump_pre_encoder_output or \
                dump_reconstructed_output or \
                dump_predicted_phenos or \
                dump_predicted_signatures:
            self.logger.dump_latent_to_csv(controller_output=controller_ret,
                                           dump_main=test_main & dump_latent,
                                           dump_lat_pre=dump_pre_encoder_output,
                                           dump_re_x=dump_reconstructed_output, re_x_col_naming=reconstructed_output_naming,
                                           dump_pheno=test_pheno & dump_latent, selected_pheno=self.selected_pheno,
                                           dump_signature=test_signature & dump_latent, selected_signature=self.selected_signature,
                                           dump_pheno_out=test_pheno & dump_predicted_phenos,
                                           dump_signature_out=test_signature & dump_predicted_signatures,
                                           rownames=self.count_data.gene_expr_mat.columns[selected_split_mask],
                                           colnames=self.count_data.gene_expr_mat.index,
                                           path=csv_path,
                                           compression=compression
                                           )


    def __lint_split_configs(self, split_configs):
        # Lint split_configs
        for cur_split_key in split_configs.keys():
            if split_configs[cur_split_key]['train_pheno'] == 'True':
                if split_configs[cur_split_key].get('selected_pheno') is None:
                    split_configs[cur_split_key]['selected_pheno'] = {idx: {'loss': '*', 'regularization': '*'} for idx
                                                                      in self.selected_pheno}
                    warnings.warn(
                        "(To silence, specify phenotype selection explicitly in the config file.) Selecting all included phenotypes and linked losses and regularizations:" + str(
                            split_configs[cur_split_key]['selected_pheno']))
            else:
                if split_configs[cur_split_key].get('selected_pheno') is not None:
                    raise ValueError(
                        "Inconsistent training specification, specified phenotype to include in training but surpressed phenotype training.")

            if split_configs[cur_split_key]['train_signature'] == 'True':
                if split_configs[cur_split_key].get('selected_signature') is None:
                    split_configs[cur_split_key]['selected_signature'] = {idx: {'loss': '*', 'regularization': '*'} for
                                                                          idx in self.selected_signature}
                    warnings.warn(
                        "(To silence, specify signature selection explicitly in the config file.) Selecting all included signatures and linked losses and regularizations: " + str(
                            split_configs[cur_split_key]['selected_signature']))
            else:
                if split_configs[cur_split_key].get('selected_signature') is not None:
                    raise ValueError(
                        "Inconsistent training specification, specified signature to include in training but surpressed signature training.")
            if split_configs[cur_split_key].get('batch_size') is None:
                warnings.warn(
                    "(To slience, specify batch_size explicitly in the config file.) Using default batch_size 50.")
                split_configs[cur_split_key]['batch_size'] = 50
            return split_configs

    def train_hybrid(self, split_configs: dict, ticks=50000,
                     hybrid_mode='interleave',
                     prog_loss_weight_mode='epoch_end',
                     make_logs=True, log_prefix='', log_loss_groups=['loss', 'regularization'], save_raw_loss=False,
                     perform_test=False, test_segmant=2000, tests: dict = None,
                     perform_checkpoint=False, checkpoint_segment=2000, checkpoint_prefix='', checkpoint_save_arch=False,
                     loss_prog_on_test: dict = None,
                     resume=False, resume_dict=None):
        """
        Train model in hydrid mode

        Note: when epoch loss progressing is on, then the progression will incur only for selected loss when an epoch ends (tick reach end)

        :param split_configs:
        :param ticks:
        :param hybrid_mode:
        :param prog_loss_weight_mode:
        :param make_logs:
        :param log_prefix:
        :param perform_test:
        :param test_segmant:
        :param tests:
        :param loss_prog_on_test:
        :return:
        """
        split_configs = self.__lint_split_configs(split_configs)

        # Resume flag
        to_resume_flag = resume

        # Setup splits
        split_masks = dict()  # Masks for each splits
        split_samplers = dict()  # Samplers
        split_iters = dict()  # Dictionary of actual indices (persisted as list to support resume)
        split_idx_idx = dict()  # Index of index for each split

        if resume and to_resume_flag:
            # Resume samplers and idx_idx (automatically progressed to next)
            split_samplers = resume_dict['split_samplers']
            split_iters = resume_dict['split_iters']
            split_idx_idx = resume_dict['split_idx_idx']
            for cur_split_key in split_idx_idx.keys():
                split_masks[cur_split_key] = self.splits[split_configs[cur_split_key]['use_split']]
        else:
            # Normally, idx_idx will reset to 0, all samplers start from new
            for cur_split_key in split_configs.keys():
                split_idx_idx[cur_split_key] = 0
                split_masks[cur_split_key] = self.splits[split_configs[cur_split_key]['use_split']]
                split_samplers[cur_split_key] = torch.utils.data.BatchSampler(
                    sampler=torch.utils.data.SubsetRandomSampler(
                        np.arange(len(self.count_data))[split_masks[cur_split_key]]
                    ),
                    batch_size=split_configs[cur_split_key].get('batch_size', 50),
                    drop_last=False
                )
            # Generate and persist batches
            for cur_split_key in split_configs.keys():
                split_iters[cur_split_key] = list()
                for cur_idx in iter(split_samplers[cur_split_key]):
                    split_iters[cur_split_key].append(cur_idx)

        if hybrid_mode == 'interleave':
            # Round robin all given splits one by one
            cur_split_idx = 0
            cur_tick = 0

            # Handle resume
            if resume and to_resume_flag:
                # When resume, proceed to next split idx and tick
                cur_split_idx = resume_dict['cur_split_idx']
                cur_tick = resume_dict['cur_tick']
                to_resume_flag = False

            while cur_tick < ticks:

                # Select split
                if cur_split_idx >= len(split_configs):
                    cur_split_idx = 0
                cur_split_key = list(split_configs.keys())[cur_split_idx]

                if self.verbose:
                    logger.debug("Hybrid tick {}: {}", cur_tick, cur_split_key)
                    logger.debug("split_idx_idx: {}", split_idx_idx)

                # Make batch and maintain iterator
                if split_idx_idx[cur_split_key] >= len(split_iters[cur_split_key]):
                    # Regenerate index is required
                    split_iters[cur_split_key] = list()
                    for cur_idx in iter(split_samplers[cur_split_key]):
                        split_iters[cur_split_key].append(cur_idx)
                    split_idx_idx[cur_split_key] = 0

                    # Verbose logging
                    if self.verbose:
                        logger.debug('{} reached the end out epoch.', cur_split_key)

                    # Progress epoch
                    if prog_loss_weight_mode == 'epoch_end':
                        self.controller.next_epoch(prog_main=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                   prog_pheno=(split_configs[cur_split_key]['train_pheno'] == 'True'),
                                                   selected_pheno=split_configs[cur_split_key].get('selected_pheno'),
                                                   prog_signature=(split_configs[cur_split_key]['train_signature'] == 'True'),
                                                   selected_signature=split_configs[cur_split_key].get('selected_signature'))

                cur_idx = split_iters[cur_split_key][split_idx_idx[cur_split_key]]
                cur_batch = self.count_data[cur_idx]

                if self.verbose:
                    logger.debug("cur_batch['cell_key'][:10]: {}", cur_batch['cell_key'][:10])

                # Train
                controller_ret = self.controller.train(batch=cur_batch,
                                                       backward_reconstruction_loss=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                       backward_main_latent_regularization=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                       backward_pheno_loss=(split_configs[cur_split_key]['train_pheno'] == 'True'),
                                                       selected_pheno=split_configs[cur_split_key].get('selected_pheno'),
                                                       backward_signature_loss=(split_configs[cur_split_key]['train_signature'] == 'True'),
                                                       selected_signature=split_configs[cur_split_key].get('selected_signature'),
                                                       detach=(split_configs[cur_split_key].get('detach') == 'True'),
                                                       detach_from=split_configs[cur_split_key].get('detach_from', ''),
                                                       save_raw_loss=save_raw_loss)

                # Log
                if make_logs:
                    self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                         loss_name_prefix=log_prefix, selected_loss_group=log_loss_groups)

                # Proceed to next tick, tick controller (so that when resume, the model has already been on next tick)
                self.controller.tick()
                cur_split_idx += 1
                cur_tick += 1
                split_idx_idx[cur_split_key] += 1

                # Persist split index and tick for resume
                resume_dict['cur_split_idx'] = cur_split_idx
                resume_dict['cur_tick'] = cur_tick
                resume_dict['split_idx_idx'] = split_idx_idx
                resume_dict['split_samplers'] = split_samplers
                resume_dict['split_iters'] = split_iters

                # Test
                if perform_test and cur_tick % test_segmant == 0:
                    # Perform test
                    for cur_test in tests:
                        # When test, all latents will be evaluated
                        self.test(split_id=cur_test['on_split'],
                                  test_main=not (cur_test.get('test_main') == 'False'),
                                  test_pheno=not (cur_test.get('test_pheno') == 'False'), selected_pheno=cur_test.get('selected_pheno'),
                                  test_signature=not (cur_test.get('test_signature') == 'False'), selected_signature=cur_test.get('selected_signature'),
                                  make_logs=(cur_test.get('make_logs') == 'True'),
                                  log_loss_groups=cur_test.get('log_loss_groups', log_loss_groups),
                                  log_prefix=cur_test.get('log_prefix', log_prefix),
                                  dump_latent=(cur_test.get('dump_latent') == 'True'),
                                  latent_prefix=cur_test.get('latent_prefix', ''),
                                  save_raw_loss=save_raw_loss)

                    if prog_loss_weight_mode == 'on_test':
                        self.controller.next_epoch(prog_main=(loss_prog_on_test['prog_main'] == 'True'),
                                                   prog_pheno=(loss_prog_on_test['train_pheno'] == 'True'),
                                                   selected_pheno=loss_prog_on_test.get('selected_pheno'),
                                                   prog_signature=(loss_prog_on_test['train_signature'] == 'True'),
                                                   selected_signature=loss_prog_on_test.get('selected_signature'))
                # Checkpoint
                if perform_checkpoint and cur_tick % checkpoint_segment == 0:
                    # Save checkpoint
                    self.save_checkpoint(training_state=resume_dict,
                                         checkpoint_path=self.log_path + checkpoint_prefix + '_tick_' + str(cur_tick) + '.pth',
                                         save_model_arch=checkpoint_save_arch)




        elif hybrid_mode == 'pattern':
            # TODO: Progress batches by given pattern (or even more advanced, interleaving + sum mixturn, seems not so useful currently)
            raise NotImplementedError
        elif hybrid_mode == 'sum':
            # Forward all splits simultaneously, then sum the loss and backward altogether
            cur_tick = 0

            # Handle resume
            if resume and to_resume_flag:
                cur_tick = resume_dict['cur_tick']
                to_resume_flag = False

            while cur_tick < ticks:
                # Verbose loggings
                if self.verbose:
                    logger.debug("Hybrid tick {}: summing loss from {}", cur_tick, list(split_configs.keys()))

                # Tensor for collecting all losses
                cur_losses = dict()
                total_loss = torch.Tensor([0.])
                if self.device == 'cuda':
                    total_loss.cuda()

                for cur_split_key in split_configs.keys():
                    # Make batch and maintain iterator
                    if split_idx_idx[cur_split_key] >= len(split_iters[cur_split_key]):
                        # Regenerate index is required
                        split_iters[cur_split_key] = list()
                        for cur_idx in iter(split_samplers[cur_split_key]):
                            split_iters[cur_split_key].append(cur_idx)
                        split_idx_idx[cur_split_key] = 0

                        # Verbose logging
                        if self.verbose:
                            logger.debug("{} reached the end out epoch.", cur_split_key)

                        # Progress epoch
                        if prog_loss_weight_mode == 'epoch_end':
                            self.controller.next_epoch(prog_main=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                       prog_pheno=(split_configs[cur_split_key]['train_pheno'] == 'True'),
                                                       selected_pheno=split_configs[cur_split_key].get('selected_pheno'),
                                                       prog_signature=(split_configs[cur_split_key]['train_signature'] == 'True'),
                                                       selected_signature=split_configs[cur_split_key].get('selected_signature'))

                    cur_idx = split_iters[cur_split_key][split_idx_idx[cur_split_key]]
                    cur_batch = self.count_data[cur_idx]

                    if self.verbose:
                        logger.debug("Split: {}", cur_split_key)
                        logger.debug("cur_batch['cell_key'][:10]: {}", cur_batch['cell_key'][:10])

                    # Obtain loss
                    cur_losses[cur_split_key] = self.controller.train(batch=cur_batch,
                                                                      backward_reconstruction_loss=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                                      backward_main_latent_regularization=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                                                                      backward_pheno_loss=(split_configs[cur_split_key]['train_pheno'] == 'True'),
                                                                      selected_pheno=split_configs[cur_split_key].get('selected_pheno'),
                                                                      backward_signature_loss=(split_configs[cur_split_key]['train_signature'] == 'True'),
                                                                      selected_signature=split_configs[cur_split_key].get('selected_signature'),
                                                                      suppress_backward=True,
                                                                      detach=(split_configs[cur_split_key].get('detach') == 'True'),
                                                                      detach_from=split_configs[cur_split_key].get('detach_from', ''),
                                                                      save_raw_loss=save_raw_loss)
                    total_loss += cur_losses[cur_split_key]['total_loss_backwarded']
                    # Log
                    if make_logs:
                        self.logger.log_loss(trainer_output=cur_losses[cur_split_key], tick=self.controller.cur_tick,
                                             loss_name_prefix=log_prefix, selected_loss_group=log_loss_groups)

                    # Proceed to next batch for current split
                    split_idx_idx[cur_split_key] += 1

                    # Tick controller
                    self.controller.tick()

                # Execute backward
                self.controller.optimizer.zero_grad()
                total_loss.backward()
                self.controller.optimizer.step()

                # Progress tick (of current training task) to next
                cur_tick += 1

                # Persist split index and tick for resume
                resume_dict['cur_tick'] = cur_tick
                resume_dict['split_idx_idx'] = split_idx_idx
                resume_dict['split_samplers'] = split_samplers
                resume_dict['split_iters'] = split_iters

                # Test
                if perform_test and cur_tick % test_segmant == 0:
                    # Perform test
                    for cur_test in tests:
                        # When test, all latents will be evaluated
                        self.test(split_id=cur_test['on_split'],
                                  test_main=not (cur_test.get('test_main') == 'False'),
                                  test_pheno=not (cur_test.get('test_pheno') == 'False'), selected_pheno=cur_test.get('selected_pheno'),
                                  test_signature=not (cur_test.get('test_signature') == 'False'), selected_signature=cur_test.get('selected_signature'),
                                  make_logs=(cur_test.get('make_logs') == 'True'),
                                  log_loss_groups=cur_test.get('log_loss_groups', log_loss_groups),
                                  log_prefix=cur_test.get('log_prefix', log_prefix),
                                  dump_latent=(cur_test.get('dump_latent') == 'True'),
                                  latent_prefix=cur_test.get('latent_prefix', ''),
                                  save_raw_loss=save_raw_loss)

                    if prog_loss_weight_mode == 'on_test':
                        self.controller.next_epoch(prog_main=(loss_prog_on_test['prog_main'] == 'True'),
                                                   prog_pheno=(loss_prog_on_test['train_pheno'] == 'True'),
                                                   selected_pheno=loss_prog_on_test.get('selected_pheno'),
                                                   prog_signature=(loss_prog_on_test['train_signature'] == 'True'),
                                                   selected_signature=loss_prog_on_test.get('selected_signature'))

                # Checkpoint
                if perform_checkpoint and cur_tick % checkpoint_segment == 0:
                    # Save checkpoint
                    self.save_checkpoint(training_state=resume_dict,
                                         checkpoint_path=self.log_path + checkpoint_prefix + '_tick_' + str(cur_tick) + '.pth',
                                         save_model_arch=checkpoint_save_arch)

    def train_hybrid_fastload(self, split_configs: dict, ticks=50000,
                             hybrid_mode='interleave',
                             prog_loss_weight_mode='epoch_end',
                             make_logs=True, log_prefix='', log_loss_groups=['loss', 'regularization'], save_raw_loss=False,
                             perform_test=False, test_segmant=2000, tests: dict = None,
                             perform_checkpoint=False, checkpoint_segment=2000, checkpoint_prefix='', checkpoint_save_arch=False,
                             loss_prog_on_test: dict = None,
                             resume=False, resume_dict=None,
                             prefetch_strategy='reuse', reuse_factor=8, reuse_shuffle_when_reassign=False):
        """
        Will implement the multithread dataloader version of training flowcontrol with storyline.

        :return:
        """
        logger.info("Using multi-task dataloader")

        split_configs = self.__lint_split_configs(split_configs)

        # Resume flag
        to_resume_flag = resume

        # Setup splits
        split_masks = dict()  # Masks for each splits
        split_dataloaders = dict()  # Samplers
        split_iters = dict()  # Dictionary of actual indices (persisted as list to support resume)
        split_idx_idx = dict()  # Index of index for each split

        if resume and to_resume_flag:
            raise NotImplementedError('Resuming for fast dataload training is not implemented yet.')
        else:
            # Normally, idx_idx will reset to 0, all samplers start from new
            for cur_split_key in split_configs.keys():
                split_idx_idx[cur_split_key] = 0
                split_masks[cur_split_key] = self.splits[split_configs[cur_split_key]['use_split']]

                split_dataloaders[cur_split_key] = torch.utils.data.DataLoader(
                    dataset=self.count_data,
                    batch_size=split_configs[cur_split_key].get('batch_size', 50),
                    sampler=torch.utils.data.SubsetRandomSampler(
                        np.arange(len(self.count_data))[split_masks[cur_split_key]]
                    ),
                    drop_last=True, # Drop last is enabled here to prevent possible type conversion errors produced by the trialing dust batch
                    num_workers=split_configs[cur_split_key].get('dataloader_num_workers', 20),
                    collate_fn=self.count_data.collate_fn,
                    pin_memory=False,
                    persistent_workers=True,
                    prefetch_factor=2
                )

        prefetch_iters = dict()
        logger.info("Prefetching batches")
        if prefetch_strategy == 'reuse':
            init_prefetch_multiplier = reuse_factor
            # Generate and persist batches
            for cur_split_key in split_configs.keys():
                prefetch_iters[cur_split_key] = dict()
                for cur_prefetch_round in range(init_prefetch_multiplier):
                    prefetch_iters[cur_split_key][cur_prefetch_round] = list()
                    if self.verbose:
                        logger.debug('Prefetching batches for split: {}, round {}', cur_split_key, cur_prefetch_round)
                    for cur_i, cur_batch in enumerate(tqdm(split_dataloaders[cur_split_key])):
                        prefetch_iters[cur_split_key][cur_prefetch_round].append(deepcopy(cur_batch))
                    split_idx_idx[cur_split_key] = 0
                split_iters[cur_split_key] = prefetch_iters[cur_split_key][0]
        elif prefetch_strategy == 'fresh':
            for cur_split_key in split_configs.keys():
                split_iters[cur_split_key] = list()
                if self.verbose:
                    logger.debug('Prefetching batches for split: {}', cur_split_key)
                for cur_i, cur_batch in enumerate(tqdm(split_dataloaders[cur_split_key])):
                    split_iters[cur_split_key].append(deepcopy(cur_batch))
                split_idx_idx[cur_split_key] = 0
        logger.success("Prefetched batches")

        if hybrid_mode == 'interleave':
            # Round robin all given splits one by one
            cur_split_idx = 0
            cur_tick = 0

            # Handle resume
            if resume and to_resume_flag:
                raise NotImplementedError('Resuming for fast dataload training is not implemented yet.')

            while cur_tick < ticks:

                # Turn dataset to key-only mode to make batch generation faster
                self.count_data.mode = 'key'

                # Select split
                if cur_split_idx >= len(split_configs):
                    cur_split_idx = 0
                cur_split_key = list(split_configs.keys())[cur_split_idx]

                if self.verbose:
                    logger.debug("Hybrid tick {}: {}", cur_tick, cur_split_key)
                    logger.debug("split_idx_idx {}", split_idx_idx)

                # Make batch and maintain iterator
                if split_idx_idx[cur_split_key] >= len(split_iters[cur_split_key]):
                    if self.verbose:
                        logger.debug("Reached epoch-end: {}", cur_split_key)
                    if prefetch_strategy == 'fresh':
                        # Regenerate index is required
                        split_iters[cur_split_key] = list()
                        if self.verbose:
                            logger.debug('Prefetching batches for split: {}', cur_split_key)
                        for cur_i, cur_batch in enumerate(tqdm(split_dataloaders[cur_split_key])):
                            split_iters[cur_split_key].append(deepcopy(cur_batch))
                    elif prefetch_strategy == 'reuse':
                        # Reassign current batch iter
                        next_iter_key = random.randint(0, len(prefetch_iters[cur_split_key])-1)
                        if self.verbose:
                            logger.debug('Reassigned prefetched batches for split: {} to group {}', cur_split_key, next_iter_key)
                        split_iters[cur_split_key] = prefetch_iters[cur_split_key][next_iter_key]
                        if reuse_shuffle_when_reassign:
                            random.shuffle(split_iters[cur_split_key])
                    split_idx_idx[cur_split_key] = 0

                    # Progress epoch
                    if prog_loss_weight_mode == 'epoch_end':
                        self.controller.next_epoch(
                            prog_main=(split_configs[cur_split_key]['train_main_latent'] == 'True'),
                            prog_pheno=(split_configs[cur_split_key]['train_pheno'] == 'True'),
                            selected_pheno=split_configs[cur_split_key].get('selected_pheno'),
                            prog_signature=(split_configs[cur_split_key]['train_signature'] == 'True'),
                            selected_signature=split_configs[cur_split_key].get('selected_signature'))

                cur_batch = split_iters[cur_split_key][split_idx_idx[cur_split_key]]

                if self.verbose:
                    logger.debug("cur_batch['cell_key'][:10]: {}", cur_batch['cell_key'][:10])

                # Train
                controller_ret = self.controller.train(batch=cur_batch,
                                                       backward_reconstruction_loss=(split_configs[cur_split_key][
                                                                                         'train_main_latent'] == 'True'),
                                                       backward_main_latent_regularization=(
                                                                   split_configs[cur_split_key][
                                                                       'train_main_latent'] == 'True'),
                                                       backward_pheno_loss=(split_configs[cur_split_key][
                                                                                'train_pheno'] == 'True'),
                                                       selected_pheno=split_configs[cur_split_key].get(
                                                           'selected_pheno'),
                                                       backward_signature_loss=(split_configs[cur_split_key][
                                                                                    'train_signature'] == 'True'),
                                                       selected_signature=split_configs[cur_split_key].get(
                                                           'selected_signature'),
                                                       detach=(split_configs[cur_split_key].get('detach') == 'True'),
                                                       detach_from=split_configs[cur_split_key].get('detach_from', ''),
                                                       save_raw_loss=save_raw_loss)

                # Log
                if make_logs:
                    self.logger.log_loss(trainer_output=controller_ret, tick=self.controller.cur_tick,
                                         loss_name_prefix=log_prefix, selected_loss_group=log_loss_groups)

                # Proceed to next tick, tick controller (so that when resume, the model has already been on next tick)
                self.controller.tick()
                cur_split_idx += 1
                cur_tick += 1
                split_idx_idx[cur_split_key] += 1


                # Test
                if perform_test and cur_tick % test_segmant == 0:
                    # Before performing test, turn dataset to full mode
                    self.count_data.mode = 'all'
                    # Perform test
                    for cur_test in tests:
                        # When test, all latents will be evaluated
                        self.test(split_id=cur_test['on_split'],
                                  test_main=not (cur_test.get('test_main') == 'False'),
                                  test_pheno=not (cur_test.get('test_pheno') == 'False'),
                                  selected_pheno=cur_test.get('selected_pheno'),
                                  test_signature=not (cur_test.get('test_signature') == 'False'),
                                  selected_signature=cur_test.get('selected_signature'),
                                  make_logs=(cur_test.get('make_logs') == 'True'),
                                  log_loss_groups=cur_test.get('log_loss_groups', log_loss_groups),
                                  log_prefix=cur_test.get('log_prefix', log_prefix),
                                  dump_latent=(cur_test.get('dump_latent') == 'True'),
                                  latent_prefix=cur_test.get('latent_prefix', ''),
                                  save_raw_loss=save_raw_loss)

                    if prog_loss_weight_mode == 'on_test':
                        self.controller.next_epoch(prog_main=(loss_prog_on_test['prog_main'] == 'True'),
                                                   prog_pheno=(loss_prog_on_test['train_pheno'] == 'True'),
                                                   selected_pheno=loss_prog_on_test.get('selected_pheno'),
                                                   prog_signature=(loss_prog_on_test['train_signature'] == 'True'),
                                                   selected_signature=loss_prog_on_test.get('selected_signature'))
                # Checkpoint
                if perform_checkpoint and cur_tick % checkpoint_segment == 0:
                    # Save checkpoint
                    self.save_checkpoint(training_state=resume_dict,
                                         checkpoint_path=self.log_path + checkpoint_prefix + '_tick_' + str(
                                             cur_tick) + '.pth',
                                         save_model_arch=checkpoint_save_arch)
        else:
            raise NotImplementedError

    def save_checkpoint(self, training_state=None,
                        checkpoint_path=None,
                        save_model_arch=False, save_config=False):
        # Save model by controllor
        controllor_ret = self.controller.save_checkpoint(save_model_arch=save_model_arch,
                                                         save_config=save_config)
        # Merge training state with controller_ret
        controllor_ret['training_state'] = training_state

        # Random state
        controllor_ret['torch_rng_state'] = torch.get_rng_state()
        controllor_ret['numpy_random_state'] = np.random.get_state()
        controllor_ret['random_state'] = random.getstate()

        torch.save(controllor_ret, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.controller.load_checkpoint(state_dict=checkpoint)

        # Random states
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        random.setstate(checkpoint['random_state'])

        return checkpoint

    def execute_inference(self, story: list, resume_dict=None):
        if self.verbose:
            logger.debug('Executing inference routine {}', story)

        # No need to handle resume here, since there will be no training

        for cur_story_item in story:
            if self.verbose:
                logger.debug('Performing inference on story: {}', cur_story_item.get('remark'))
                logger.debug('Story item: {}', cur_story_item)
            cur_action = cur_story_item.get('action', 'test')

            if cur_action == 'test':
                # Test model on selected dataset
                self.test(split_id=cur_story_item['on_split'],
                          test_main=(cur_story_item.get('test_main') == 'True'),
                          test_pheno=(cur_story_item.get('test_pheno') == 'True'), selected_pheno=cur_story_item.get('selected_pheno'),
                          test_signature=(cur_story_item.get('test_signature') == 'True'), selected_signature=cur_story_item.get('selected_signature'),
                          make_logs=(cur_story_item.get('make_logs') == 'True'), log_prefix=cur_story_item.get('log_prefix', 'inference'),
                          dump_latent=(cur_story_item.get('dump_latent') == 'True'), latent_prefix=cur_story_item.get('latent_prefix', ''),
                          dump_pre_encoder_output=(cur_story_item.get('dump_pre_encoder_output') == 'True'),
                          dump_reconstructed_output=(cur_story_item.get('dump_reconstructed_output') == 'True'), reconstructed_output_naming=cur_story_item.get('reconstructed_output_naming'),
                          dump_predicted_phenos=(cur_story_item.get('dump_predicted_phenos') == 'True'),
                          dump_predicted_signatures=(cur_story_item.get('dump_predicted_signatures') == 'True'),
                          compression=(cur_story_item.get('compression', 'none')),
                          save_raw_loss=(cur_story_item.get('save_raw_loss') == 'True'))


    def train_story(self, story: list,
                    resume=False, resume_dict=None):
        # Handle resume
        cur_story_item_idx = 0
        if resume:
            cur_story_item_idx = resume_dict['cur_story_item_idx']
            if self.verbose:
                logger.debug("To resume training from story indexed as {}", cur_story_item_idx)
        else:
            resume_dict = dict()

        for cur_story_item in story[cur_story_item_idx:]:
            # Verbose logging
            if self.verbose:
                logger.debug("Training story: {}", cur_story_item.get('remark'))
                logger.debug("Stroty item: {}", cur_story_item)
            cur_action = cur_story_item.get('action', 'train')

            # Persist training story info for saving checkpoints
            resume_dict['cur_story_item_idx'] = cur_story_item_idx

            if cur_action == 'train':

                # Train model in ordinary mode
                self.train(split_id=cur_story_item['use_split'],
                           train_main=(cur_story_item['train_main_latent'] == 'True'),
                           train_pheno=(cur_story_item['train_pheno'] == 'True'),
                           train_signature=(cur_story_item['train_signature'] == 'True'),
                           selected_pheno=cur_story_item.get('selected_pheno'),
                           selected_signature=cur_story_item.get('selected_signature'),
                           epoch=cur_story_item['epochs'],
                           make_logs=(cur_story_item.get('make_logs') == 'True'),
                           log_prefix=cur_story_item.get('log_prefix', 'train'),
                           log_loss_groups=cur_story_item.get('log_loss_groups', ['loss', 'regularization']),
                           batch_size=cur_story_item.get('batch_size'),
                           test_every_epoch=(cur_story_item.get('test_every_epoch') == 'True'),
                           tests=cur_story_item.get('tests'),
                           test_on_segment=(cur_story_item.get('test_on_segment') == 'True'),
                           test_segment=cur_story_item.get('test_segment'),
                           checkpoint_on_segment=(cur_story_item.get('checkpoint_on_segment') == 'True'),
                           checkpoint_segment=(cur_story_item.get('checkpoint_segment', 2000)),
                           checkpoint_prefix=(cur_story_item.get('checkpoint_prefix')),
                           checkpoint_save_arch=(cur_story_item.get('checkpoint_save_arch') == 'True'),
                           resume=resume, resume_dict=resume_dict,
                           detach=(cur_story_item.get('detach') == 'True'),
                           detach_from=cur_story_item.get('detach_from', ''),
                           save_raw_loss=(cur_story_item.get('save_raw_loss') == 'True'))

                # Handle resume completion
                if resume:
                    resume = False
                    resume_dict = dict()

            elif cur_action == 'test':

                # Ignore if resume checkpoint starts here
                if resume:
                    resume = False
                    resume_dict = dict()

                # Test model
                self.test(split_id=cur_story_item['on_split'],
                          test_main=(cur_story_item.get('test_main', 'True') == 'True'),
                          test_pheno=(cur_story_item.get('test_pheno', 'True') == 'True'), selected_pheno=cur_story_item.get('selected_pheno'),
                          test_signature=(cur_story_item.get('test_signature', 'True') == 'True'), selected_signature=cur_story_item.get('selected_signature'),
                          make_logs=(cur_story_item.get('make_logs') == 'True'), log_prefix=cur_story_item.get('log_prefix', 'inference'),
                          log_loss_groups=cur_story_item.get('log_loss_groups', ['loss', 'regularization']),
                          dump_latent=(cur_story_item.get('dump_latent') == 'True'), latent_prefix=cur_story_item.get('latent_prefix', ''),
                          dump_pre_encoder_output=(cur_story_item.get('dump_pre_encoder_output') == 'True'),
                          dump_reconstructed_output=(cur_story_item.get('dump_reconstructed_output') == 'True'), reconstructed_output_naming=cur_story_item.get('reconstructed_output_naming'),
                          dump_predicted_phenos=(cur_story_item.get('dump_predicted_phenos') == 'True'),
                          dump_predicted_signatures=(cur_story_item.get('dump_predicted_signatures') == 'True'),
                          compression=(cur_story_item.get('compression', 'none')),
                          save_raw_loss=(cur_story_item.get('save_raw_loss') == 'True'))

            elif cur_action == 'train_hybrid':

                # Verbose logging
                if self.verbose:
                    logger.debug("Current story invokes hybrid training")

                # Train model in hybrid mode
                self.train_hybrid(split_configs=cur_story_item['split_configs'],
                                  ticks=cur_story_item.get('ticks'),
                                  hybrid_mode=cur_story_item.get('hybrid_mode'),
                                  prog_loss_weight_mode=cur_story_item.get('prog_loss_weight_mode'),
                                  make_logs=(cur_story_item.get('make_logs') == 'True'),
                                  log_prefix=cur_story_item.get('log_prefix', ''),
                                  log_loss_groups=cur_story_item.get('log_loss_groups', ['loss', 'regularization']),
                                  perform_test=(cur_story_item.get('perform_test') == 'True'),
                                  test_segmant=cur_story_item.get('test_segment'),
                                  tests=cur_story_item.get('tests'),
                                  loss_prog_on_test=cur_story_item.get('loss_prog_on_test'),
                                  perform_checkpoint=(cur_story_item.get('perform_checkpoint') == 'True'),
                                  checkpoint_segment=(cur_story_item.get('checkpoint_segment', 2000)),
                                  checkpoint_prefix=(cur_story_item.get('checkpoint_prefix')),
                                  checkpoint_save_arch=(cur_story_item.get('checkpoint_save_arch') == 'True'),
                                  resume=resume, resume_dict=resume_dict, save_raw_loss=(cur_story_item.get('save_raw_loss') == 'True'))

                # Handle resume completion
                if resume:
                    resume = False
                    resume_dict = dict()

            elif cur_action == 'train_hybrid_fastload':
                torch.multiprocessing.set_sharing_strategy('file_system')
                count_data_mode_before = self.count_data.mode
                self.count_data.mode = 'key'

                # Verbose logging
                if self.verbose:
                    logger.debug("Current story invokes hybrid training with fastload")
                    logger.debug('Set torch.multiprocessing.set_sharing_strategy: file_system')
                    logger.debug('Temporarily Set count_data.mode: key')


                # Train model in hybrid mode
                self.train_hybrid_fastload(split_configs=cur_story_item['split_configs'],
                                  ticks=cur_story_item.get('ticks'),
                                  hybrid_mode=cur_story_item.get('hybrid_mode'),
                                  prog_loss_weight_mode=cur_story_item.get('prog_loss_weight_mode'),
                                  make_logs=(cur_story_item.get('make_logs') == 'True'),
                                  log_prefix=cur_story_item.get('log_prefix', ''),
                                  log_loss_groups=cur_story_item.get('log_loss_groups', ['loss', 'regularization']),
                                  perform_test=(cur_story_item.get('perform_test') == 'True'),
                                  test_segmant=cur_story_item.get('test_segment'),
                                  tests=cur_story_item.get('tests'),
                                  loss_prog_on_test=cur_story_item.get('loss_prog_on_test'),
                                  perform_checkpoint=(cur_story_item.get('perform_checkpoint') == 'True'),
                                  checkpoint_segment=(cur_story_item.get('checkpoint_segment', 2000)),
                                  checkpoint_prefix=(cur_story_item.get('checkpoint_prefix')),
                                  checkpoint_save_arch=(cur_story_item.get('checkpoint_save_arch') == 'True'),
                                  resume=resume, resume_dict=resume_dict, save_raw_loss=(cur_story_item.get('save_raw_loss') == 'True'),
                                  prefetch_strategy=(cur_story_item.get('prefetch_strategy', 'fresh')),
                                  reuse_factor=(cur_story_item.get('reuse_factor', 5)),
                                  reuse_shuffle_when_reassign=(cur_story_item.get('reuse_shuffle_when_reassign') == 'True'))

                self.count_data.mode = count_data_mode_before

                # Handle resume completion
                if resume:
                    resume = False
                    resume_dict = dict()

            cur_story_item_idx += 1


    def insert_external_module(self, insert_config:dict, verbose=True):
        if verbose:
            logger.debug('Inserting external modules...')
            logger.debug('Insertion config: {}', insert_config)

        for cur_item_i in insert_config:
            cur_insert_item = insert_config[cur_item_i]

            # Load source model configs
            ext_model_config_path = cur_insert_item['ext_model_config_path']
            with open(ext_model_config_path, 'r') as f:
                ext_model_config = json.load(f)
            ext_signature_config_path = cur_insert_item['ext_signature_config_path']
            with open(ext_signature_config_path, 'r') as f:
                ext_signature_config = json.load(f)
            ext_pheno_config_path = cur_insert_item['ext_pheno_config_path']
            with open(ext_pheno_config_path, 'r') as f:
                ext_pheno_config = json.load(f)

            # Load saved model parameter
            ext_checkpoint_path = cur_insert_item['ext_checkpoint_path']
            ext_checkpoint = torch.load(ext_checkpoint_path)

            # Resume model
            ## Use the input dimension of pre-encoder to probe input dimension
            probed_model_input_dim = ext_checkpoint['model_state_dict']['model.pre_encoder.model_list.0.weight'].shape[1]
            ext_model = Extractor(input_dim=probed_model_input_dim,
                                  signature_config=ext_signature_config,
                                  pheno_config=ext_pheno_config,
                                  main_lat_config=ext_model_config['main_latent'],
                                  pre_encoder_config=ext_model_config.get('pre_encoder_config'),
                                  verbose=verbose)

            ext_model.load_state_dict(ext_checkpoint['model_state_dict'])
            if verbose:
                logger.debug('{}: External model resumed', cur_item_i)

            source_model = None
            if cur_insert_item.get('source') == 'decoder':
                source_model = deepcopy(ext_model.model['decoder'])
            elif cur_insert_item.get('source') == 'pheno_models':
                # TODO: to allow inserting other pre-trained modules, e.g. classifiers, regressors
                # generally, the latent dim should be consistent
                # may also implement adaptor layer for extra transformations
                source_model = deepcopy(ext_model.model['pheno_models'][cur_insert_item.get('source_name')])
            elif cur_insert_item.get('source') == 'signature_regressors':
                source_model = deepcopy(ext_model.model['signature_regressors'][cur_insert_item.get('source_name')])
            else:
                raise NotImplementedError('Insertion type is not specified')

            if self.verbose:
                logger.debug('{}: Loaded {} {} from the external model', cur_item_i,  cur_insert_item.get('source'), cur_insert_item.get('source_name', ''))

            if cur_insert_item.get('destination_type') == 'decoder':
                self.model.model['decoder'] = source_model
                if self.verbose:
                    logger.debug('{}: Replaced current decoder with loaded external module', cur_item_i)
            elif cur_insert_item.get('destination_type') == 'pheno':
                self.model.model['pheno_models'][cur_insert_item.get('destination_name')] = source_model
                if self.verbose:
                    logger.debug('{}: Replaced pheno model {} with loaded external module', cur_item_i, cur_insert_item.get('destination_name'))
            elif cur_insert_item.get('destination_type') == 'signature':
                self.model.model['signature_regressors'][cur_insert_item.get('destination_name')] = source_model
                if self.verbose:
                    logger.debug('{}: Replaced signature model {} with loaded external module',cur_item_i , cur_insert_item.get('destination_name'))

@logger.catch
def main():
    logger.info('SCARE/SAKRA Prototype')
    logger.info('Working directory: {}', os.getcwd())

    args = parse_args()
    if type(args.inference) is str and len(args.inference) > 0:
        if args.verbose:
            logger.debug("Entering inference mode from checkpoint: {}", args.inference)

        # Load inference story file
        if args.verbose:
            logger.debug("Loading inference story file: {}", args.inference_story)

        # Load inference story *list*
        with open(args.inference_story, 'r') as f:
            inference_story = json.load(f)
        if type(inference_story) is dict:
            inference_story = [inference_story[x] for x in inference_story]

        # Init SAKRA instance
        instance = SAKRA(config_json_path=args.config,
                         verbose=args.verbose,
                         suppress_train=True,
                         suppress_tensorboardX=args.suppress_tensorboardX)

        # Load session
        checkpoint = instance.load_checkpoint(args.inference)

        # Execute specified inference routine
        instance.execute_inference(story=inference_story)


    elif type(args.resume) is str and len(args.resume) > 0:
        if args.verbose:
            logger.info("Resuming checkpoint:", args.resume)

        # Init SAKRA instance
        instance = SAKRA(config_json_path=args.config,
                         verbose=args.verbose,
                         suppress_train=True)

        # Load session
        checkpoint = instance.load_checkpoint(args.resume)

        # Resume training
        instance.train_story(story=instance.config['story'],
                             resume=True,
                             resume_dict=checkpoint['training_state'])
    elif args.external_module:
        if args.verbose:
            logger.info("Merging external modules")

        # Error check: should not simultaneously resume/inference while merging with external modules
        if type(args.inference) is str and len(args.inference) > 0:
            raise ValueError(
                'Do not perform inference when importing external modules, try to import and save a merged model first')
        if type(args.resume) is str and len(args.resume) > 0:
            raise ValueError(
                'Do not resume training when importing external modules, try to import and save a merged model first')

        # Init SAKURA instance (suppress training to merge modules)
        instance = SAKRA(config_json_path=args.config,
                         verbose=args.verbose,
                         suppress_train=True)

        # Load insert config
        insert_config_path = args.external_module_path
        with open(insert_config_path, 'r') as f:
            insert_config = json.load(f)

        # Merge external model
        instance.insert_external_module(insert_config=insert_config,
                                        verbose=args.verbose)

        # Re-init optimizer
        instance.controller.setup_optimizer()

        # Train as normal
        instance.train_story(story=instance.config['story'])


    else:
        instance = SAKRA(config_json_path=args.config,
                         verbose=args.verbose,
                         suppress_train=args.suppress_train)

if __name__ == '__main__':
    main()
