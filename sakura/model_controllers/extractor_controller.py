import functools

import torch
import torch.linalg
import torch.nn.functional
import torch.optim
from tabulate import tabulate

import sakura.utils.distributions as distributions
from sakura.models.extractor import Extractor
from sakura.utils.gradient_reverse import NeutralizeLayerF
from sakura.utils.gradient_reverse import ReverseLayerF
from sakura.utils.sliced_wasserstein import SlicedWasserstein
from sakura.utils.kl_divergence import KLDivergence


class ExtractorController(object):
    def __init__(self, model: Extractor,
                 config: dict = None,
                 pheno_config: dict = None,
                 signature_config: dict = None,
                 verbose=False):

        """
        Initializer of Extractor controller.
        :param model: Extractor model
        :param config: experiment configuration dict
        :param pheno_config: phenotype supervision configuration
        :param signature_config: signature supervision configuration
        :param verbose: should verbose console output/logging
        """

        self.verbose = verbose

        self.model = model
        self.config = config
        self.device = self.config['device']

        # SW2 regularizer and defaults
        self.SW2 = SlicedWasserstein()
        self.KL = KLDivergence()
        self.main_latent_config = self.config['main_latent']

        # Line main latent config
        # Supporting no regularization option on main latent space
        if self.main_latent_config.get('regularization') is None:
            self.main_latent_config['regularization'] = {}
        # Requiring loss for reconstruction
        if self.main_latent_config.get('loss') is None:
            raise ValueError('At least one loss should be configured for main latent space')

        # Phenotype supervision configs (internal)
        if pheno_config is None:
            pheno_config = dict()
        self.pheno_config = pheno_config
        # Lint phenotype configs
        for cur_pheno in self.pheno_config.keys():
            # Requiring loss to be configured
            if self.pheno_config[cur_pheno].get('loss') is None:
                raise ValueError('No loss configured for phenotype:' + cur_pheno)
            # Optional regularization
            if self.pheno_config[cur_pheno].get('regularization') is None:
                self.pheno_config[cur_pheno]['regularization'] = {}

        # Signature supervision configs (internal)
        if signature_config is None:
            signature_config = dict()
        self.signature_config = signature_config
        # Lint signature configs
        for cur_signature in self.signature_config.keys():
            if self.signature_config[cur_signature].get('loss') is None:
                raise ValueError('No loss configured for signature:' + cur_signature)
            if self.signature_config[cur_signature].get('regularization') is None:
                self.signature_config[cur_signature]['regularization'] = {}

        # Init trainer states
        self.cur_tick = 0
        self.cur_epoch = 0
        self.main_loss_weight = dict()
        self.pheno_loss_weight = dict()
        self.signature_loss_weight = dict()
        self.reset()

        # Move model to assigned device
        # TODO: GPU support of Extractor controller
        if self.device == 'cuda':
            self.model.cuda()

        self.setup_optimizer()

        if verbose:
            print('===========================')
            print('Extractor controller')
            print(self.pheno_config)
            print(self.signature_config)
            print(self.main_latent_config)
            print(self.optimizer)
            self.print_weight_projection(expected_epoch=100)
            print('===========================')



    def __scan_exclude(self, cur_model_type=None, cur_model_name=None):
        if self.config['optimizer'].get('excludes') is None:
            if self.verbose:
                print(cur_model_type, cur_model_name, 'Passed')
            return True
        if type(self.config['optimizer'].get('excludes')) is list:
            for cur_item in self.config['optimizer'].get('excludes'):
                if self.verbose:
                    print("Checking", cur_item)
                if cur_model_type == cur_item['type']:
                    if cur_item['type'] == 'pre_encoder':
                        return False
                    elif cur_item['type'] == 'main_latent_compressor':
                        return False
                    elif cur_item['type'] == 'decoder':
                        return False
                    elif cur_item['name'] == cur_model_name:
                        return False
            if self.verbose:
                print(cur_model_type, cur_model_name, 'Passed')
            return True
        else:
            raise TypeError


    def setup_optimizer(self):
        # Filter parameter
        modules_to_optim = torch.nn.ModuleList()
        # Scan through all parameters
        for cur_model_type in self.model.model:
            if isinstance(self.model.model[cur_model_type], torch.nn.ModuleDict):
                for cur_model_name in self.model.model[cur_model_type]:
                    if self.__scan_exclude(cur_model_type, cur_model_name):
                        modules_to_optim.append(self.model.model[cur_model_type][cur_model_name])
            elif self.__scan_exclude(cur_model_type):
                modules_to_optim.append(self.model.model[cur_model_type])

        if self.verbose:
            print('Filtered parameter list:')
            print(modules_to_optim)

        # Setup Optimizer
        if self.config['optimizer']['type'] == 'RMSProp':
            self.optimizer = torch.optim.RMSprop(modules_to_optim.parameters(),
                                                 lr=self.config['optimizer']['RMSProp_lr'],
                                                 alpha=self.config['optimizer']['RMSProp_alpha'],
                                                 weight_decay=self.config['optimizer'].get('RMSProp_weight_decay', 0),
                                                 momentum=self.config['optimizer'].get('RMSProp_momentum', 0))
        else:
            print("Optimizers other than RMSProp not implemented.")
            raise NotImplementedError

    def print_weight_projection(self, expected_epoch):
        # Initial weights
        projected_weights = dict()
        for cur_group in ['loss', 'regularization']:
            for cur_main_loss_key in self.main_latent_config[cur_group].keys():
                projected_weights['main_' + cur_group + '_' + cur_main_loss_key] = [
                    self.main_latent_config[cur_group][cur_main_loss_key]['init_weight']]
        for cur_pheno in self.pheno_config.keys():
            for cur_group in ['loss', 'regularization']:
                for cur_pheno_loss_key in self.pheno_config[cur_pheno][cur_group].keys():
                    projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key] = [
                        self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['init_weight']]
        for cur_signature in self.signature_config.keys():
            for cur_group in ['loss', 'regularization']:
                for cur_signature_loss_key in self.signature_config[cur_signature][cur_group].keys():
                    projected_weights['signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key] = [
                        self.signature_config[cur_signature][cur_group][cur_signature_loss_key]['init_weight']]

        # Dynamic weights
        for cur_epoch in range(1, expected_epoch):
            for cur_group in ['loss', 'regularization']:
                for cur_main_loss_key in self.main_latent_config[cur_group].keys():
                    if cur_epoch >= self.main_latent_config[cur_group][cur_main_loss_key]['progressive_start_epoch']:
                        projected_weights['main_' + cur_group + '_' + cur_main_loss_key].append(
                            projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch - 1] *
                            self.main_latent_config[cur_group][cur_main_loss_key]['progressive_const'])
                    else:
                        projected_weights['main_' + cur_group + '_' + cur_main_loss_key].append(
                            projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch - 1])
                    if self.main_latent_config[cur_group][cur_main_loss_key].get('max_weight') is not None:
                        projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch] = min(projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch],
                                                                                                          self.main_latent_config[cur_group][cur_main_loss_key].get('max_weight'))
                    if self.main_latent_config[cur_group][cur_main_loss_key].get('min_weight') is not None:
                        projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch] = max(projected_weights['main_' + cur_group + '_' + cur_main_loss_key][cur_epoch],
                                                                                                          self.main_latent_config[cur_group][cur_main_loss_key].get('min_weight'))
            for cur_pheno in self.pheno_config.keys():
                for cur_group in ['loss', 'regularization']:
                    for cur_pheno_loss_key in self.pheno_config[cur_pheno][cur_group].keys():
                        if cur_epoch >= self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_start_epoch']:
                            projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key].append(
                                projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][
                                    cur_epoch - 1] * self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key][
                                    'progressive_const'])
                        else:
                            projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key].append(
                                projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][
                                    cur_epoch - 1])
                        if self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('max_weight') is not None:
                            projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][cur_epoch] = min(
                                projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][cur_epoch],
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('max_weight'))
                        if self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('min_weight') is not None:
                            projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][cur_epoch] = max(
                                projected_weights['pheno_' + cur_pheno + '_' + cur_group + '_' + cur_pheno_loss_key][cur_epoch],
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('min_weight'))

            for cur_signature in self.signature_config.keys():
                for cur_group in ['loss', 'regularization']:
                    for cur_signature_loss_key in self.signature_config[cur_signature][cur_group].keys():
                        if cur_epoch >= self.signature_config[cur_signature][cur_group][cur_signature_loss_key]['progressive_start_epoch']:
                            projected_weights[
                                'signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key].append(
                                projected_weights[
                                    'signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][
                                    cur_epoch - 1] *
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                    'progressive_const'])
                        else:
                            projected_weights[
                                'signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key].append(
                                projected_weights[
                                    'signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][
                                    cur_epoch - 1])
                        if self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('max_weight') is not None:
                            projected_weights['signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][cur_epoch] = min(
                                projected_weights['signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][cur_epoch],
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('max_weight'))
                        if self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('min_weight') is not None:
                            projected_weights['signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][cur_epoch] = max(
                                projected_weights['signature_' + cur_signature + '_' + cur_group + '_' + cur_signature_loss_key][cur_epoch],
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('min_weight'))

        # Make tabular and print
        table = tabulate(projected_weights, showindex='always', headers='keys')
        print(table)

    def reset(self):
        # Reset trainer state
        # Global tick and epoch counter
        self.cur_tick = 0
        self.cur_epoch = 0

        # Main latent weight
        self.main_loss_weight = {'loss': dict(), 'regularization': dict()}
        self.main_epoch = {'loss': dict(), 'regularization': dict()}
        for cur_group in ['loss', 'regularization']:
            for cur_main_loss_key in self.main_latent_config[cur_group].keys():
                self.main_loss_weight[cur_group][cur_main_loss_key] = \
                    self.main_latent_config[cur_group][cur_main_loss_key]['init_weight']
                self.main_epoch[cur_group][cur_main_loss_key] = 0

        # Phenotype supervision and regularization loss weight
        self.pheno_loss_weight = dict()
        self.pheno_epoch = dict()
        for cur_pheno in self.pheno_config.keys():
            self.pheno_loss_weight[cur_pheno] = {'loss': dict(), 'regularization': dict()}
            self.pheno_epoch[cur_pheno] = {'loss': dict(), 'regularization': dict()}
            for cur_group in ['loss', 'regularization']:
                for cur_pheno_loss_key in self.pheno_config[cur_pheno][cur_group].keys():
                    self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] = \
                        self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['init_weight']
                    self.pheno_epoch[cur_pheno][cur_group][cur_pheno_loss_key] = 0

        # Signature supervision loss weight
        self.signature_loss_weight = dict()
        self.signature_epoch = dict()
        for cur_signature in self.signature_config.keys():
            self.signature_loss_weight[cur_signature] = {'loss': dict(), 'regularization': dict()}
            self.signature_epoch[cur_signature] = {'loss': dict(), 'regularization': dict()}
            for cur_group in ['loss', 'regularization']:
                for cur_signature_loss_key in self.signature_config[cur_signature][cur_group].keys():
                    self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] = \
                        self.signature_config[cur_signature][cur_group][cur_signature_loss_key]['init_weight']
                    self.signature_epoch[cur_signature][cur_group][cur_signature_loss_key] = 0

    def tick(self):
        self.cur_tick = self.cur_tick + 1

    def next_epoch(self,
                   prog_main=True,
                   prog_pheno=True, selected_pheno=None,
                   prog_signature=True, selected_signature=None):
        """
        Handle inter epoch loss weight progressions and tick epoch counters.
        :param prog_main:
        :param prog_pheno:
        :param selected_pheno:
        :param prog_signature:
        :param selected_signature:
        :return:
        """
        # Global epoch
        self.cur_epoch = self.cur_epoch + 1

        # Main latent epoch dynamic weight
        if prog_main:
            # TODO: also implement selection of main latent loss/regularization here (not sure if useful)
            for cur_group in ['loss', 'regularization']:
                for cur_main_loss_key in self.main_latent_config[cur_group].keys():
                    self.main_epoch[cur_group][cur_main_loss_key] += 1
                    if self.main_epoch[cur_group][cur_main_loss_key] >= \
                            self.main_latent_config[cur_group][cur_main_loss_key]['progressive_start_epoch']:
                        if self.main_latent_config[cur_group][cur_main_loss_key].get('progressive_mode') is None:
                            # By default, multiply (also for back-compatibility purpose)
                            self.main_loss_weight[cur_group][cur_main_loss_key] *= \
                                self.main_latent_config[cur_group][cur_main_loss_key]['progressive_const']
                        elif self.main_latent_config[cur_group][cur_main_loss_key].get('progressive_mode') == 'multiply':
                            self.main_loss_weight[cur_group][cur_main_loss_key] *= \
                                self.main_latent_config[cur_group][cur_main_loss_key]['progressive_const']
                        elif self.main_latent_config[cur_group][cur_main_loss_key].get('progressive_mode') == 'increment':
                            self.main_loss_weight[cur_group][cur_main_loss_key] += \
                                self.main_latent_config[cur_group][cur_main_loss_key]['progressive_const']
                        else:
                            raise NotImplementedError('Dynamic weight mode not supported')
                    if self.main_latent_config[cur_group][cur_main_loss_key].get('max_weight') is not None:
                        self.main_loss_weight[cur_group][cur_main_loss_key] = min(
                            self.main_loss_weight[cur_group][cur_main_loss_key],
                            self.main_latent_config[cur_group][cur_main_loss_key].get('max_weight'))
                    if self.main_latent_config[cur_group][cur_main_loss_key].get('min_weight') is not None:
                        self.main_loss_weight[cur_group][cur_main_loss_key] = max(
                            self.main_loss_weight[cur_group][cur_main_loss_key],
                            self.main_latent_config[cur_group][cur_main_loss_key].get('min_weight'))

        # Phenotype supervision and regularization loss dynamic weight
        if prog_pheno:
            if selected_pheno is None:
                selected_pheno = {idx: {'loss': '*', 'regularization': '*'} for idx in self.pheno_config.keys()}
            for cur_pheno in selected_pheno.keys():
                for cur_group in ['loss', 'regularization']:
                    selected_pheno_loss_keys = self.select_loss_dict(
                        selection=selected_pheno[cur_pheno][cur_group],
                        internal=self.pheno_config[cur_pheno][cur_group]
                    )
                    for cur_pheno_loss_key in selected_pheno_loss_keys:
                        self.pheno_epoch[cur_pheno][cur_group][cur_pheno_loss_key] += 1
                        if self.pheno_epoch[cur_pheno][cur_group][cur_pheno_loss_key] >= \
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_start_epoch']:
                            if self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('progressive_mode') is None:
                                self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] *= \
                                    self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_const']
                            elif self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('progressive_mode') == 'multiply':
                                self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] *= \
                                    self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_const']
                            elif self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('progressive_mode') == 'increment':
                                self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] += \
                                    self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_const']
                            else:
                                raise NotImplementedError('Dynamic weight mode not supported')
                        if self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('max_weight') is not None:
                            self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] = min(
                                self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key],
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('max_weight'))
                        if self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('min_weight') is not None:
                            self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] = max(
                                self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key],
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key].get('min_weight'))

        # Signature supervision and regularization loss dynamic weight
        if prog_signature:
            if selected_signature is None:
                selected_signature = {idx: {'loss': '*', 'regularization': '*'} for idx in self.signature_config.keys()}
            for cur_signature in selected_signature.keys():
                for cur_group in ['loss', 'regularization']:
                    selected_signature_loss_keys = self.select_loss_dict(
                        selection=selected_signature[cur_signature][cur_group],
                        internal=self.signature_config[cur_signature][cur_group]
                    )
                    for cur_signature_loss_key in selected_signature_loss_keys:
                        self.signature_epoch[cur_signature][cur_group][cur_signature_loss_key] += 1
                        if self.signature_epoch[cur_signature][cur_group][cur_signature_loss_key] >= \
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                    'progressive_start_epoch']:
                            if self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('progressive_mode') is None:
                                self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] *= \
                                    self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                        'progressive_const']
                            elif self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('progressive_mode') == 'multiply':
                                self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] *= \
                                    self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                        'progressive_const']
                            elif self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get('progressive_mode') == 'increment':
                                self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] += \
                                    self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                        'progressive_const']
                            else:
                                raise NotImplementedError('Unsupported progressive mode')
                        if self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get(
                                'max_weight') is not None:
                            self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] = min(
                                self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key],
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get(
                                    'max_weight'))
                        if self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get(
                                'min_weight') is not None:
                            self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] = max(
                                self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key],
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key].get(
                                    'min_weight'))

        if self.verbose:
            print('[Extractor Controller]: next epoch, current global epoch:', self.cur_epoch)
            print('Main latent loss-wise epochs:', self.main_epoch)
            print('Main latent loss-wise weights:', self.main_loss_weight)
            print('Phenotype loss-wise epochs:', self.pheno_epoch)
            print('Phenotype loss-wise weights:', self.pheno_loss_weight)
            print('Signature loss-wise epochs:', self.signature_epoch)
            print('Signature loss-wise weights:', self.signature_loss_weight)

    def regularize(self, tensor, regularization_config: dict, supervision=None):
        """
        Handle regularizations.
        :param tensor: tensor to regularize (usually, a "batched" tensor with shape (N, ...), where N is the number of data points)
        :param regularization_config: a dict, containing regularization configuration
        :param supervision: an object required for supervised regularization
        :return:
        """
        if regularization_config['type'] == 'SW2_uniform':
            # Uniform distribution regularization
            return self.SW2(
                encoded_samples=tensor,
                distribution_fn=functools.partial(distributions.rand_uniform,
                                                  low=regularization_config['uniform_low'],
                                                  high=regularization_config['uniform_high']),
                num_projections=regularization_config['SW2_num_projections'],
                device=self.device
            )
        elif regularization_config['type'] == 'SW2_uniform_supervised':
            # Supervised uniform distribution
            return self.SW2(
                encoded_samples=tensor,
                distribution_fn=functools.partial(
                    distributions.rand_uniform,
                    low=regularization_config['uniform_low'],
                    high=regularization_config['uniform_high'],
                    n_labels=regularization_config['uniform_n_labels'],
                    label_offsets=regularization_config['uniform_label_offsets'],
                    label_indices=supervision
                ),
                num_projections=regularization_config['SW2_num_projections'],
                device=self.device
            )

        elif regularization_config['type'] == 'SW2_gaussian_mixture':
            return self.SW2(
                encoded_samples=tensor,
                distribution_fn=functools.partial(distributions.gaussian_mixture,
                                                  n_labels=regularization_config['gaussian_mixture_n_labels'],
                                                  x_var=regularization_config.get('gaussian_mixture_x_var'),
                                                  y_var=regularization_config.get('gaussian_mixture_y_var'),
                                                  label_indices=supervision),
                num_projections=regularization_config['SW2_num_projections'],
                device=self.device
            )
        elif regularization_config['type'] == 'SW2_ring2d':
            return self.SW2(
                encoded_samples=tensor,
                distribution_fn=functools.partial(distributions.rand_ring2d),
                num_projections=regularization_config['SW2_num_projections'],
                device=self.device
            )
        elif regularization_config['type'] == 'L1_regularization':
            return torch.linalg.norm(tensor, dim=1, ord=1).mean()
        elif regularization_config['type'] == 'L2_regularization':
            return torch.linalg.norm(tensor, dim=1, ord=2).mean()
        elif regularization_config['type'] == 'L0_regularization':
            # Not meaningful to use the embedded L-0 norm, it can't generate grad
            # consider: https://github.com/moskomule/l0.pytorch
            raise NotImplementedError
        elif regularization_config['type'] == 'SW2_gaussian_mixture_supervised':
            # TODO: Supervised gaussian mixture prior
            raise NotImplementedError
        elif regularization_config['type'] == 'euclidean_anchor':
            # TODO: Euclidean anchoring
            raise NotImplementedError
        elif regularization_config['type'] == 'KL_gaussian_mixture':
            return self.KL(
                encoded_samples=tensor,
                distribution_fn=functools.partial(distributions.gaussian_mixture,
                                                  n_labels=regularization_config['gaussian_mixture_n_labels'],
                                                  x_var=regularization_config.get('gaussian_mixture_x_var'),
                                                  y_var=regularization_config.get('gaussian_mixture_y_var'),
                                                  label_indices=supervision),
                device=self.device
            )


    def select_loss_dict(self, selection=None, internal=None):
        if type(selection) is list:
            return selection
        elif selection == '*':
            return internal.keys()
        else:
            raise ValueError

    def select_item_dict(self, selection=None, internal=None, mode='keys'):
        """
        Utility method to handle selections from internal phenotype/signature configs.
        :param selection: selection object, could be dict, list, str, or None.
        :param internal: internal config dict to select
        :param mode: By default, 'key', will return a key list after selection
        :return:
        """
        if mode == 'keys':
            keys_to_ret = list()
            if selection is None or selection == '*':
                # Select all in internal config
                keys_to_ret = internal.keys()
            elif type(selection) is dict:
                keys_to_ret = selection.keys()
            elif type(selection) is list:
                keys_to_ret = selection
            elif type(selection) is str:
                keys_to_ret = [selection]
            else:
                raise ValueError
            return keys_to_ret
        elif mode == 'dict':
            if selection is None or selection == '*':
                return {idx: '*' for idx in internal.keys()}
            elif type(selection) is dict():
                return selection

    def loss(self, batch, expr_key='all',
             forward_pheno=True, selected_pheno=None,
             forward_signature=True, selected_signature=None,
             forward_reconstruction=True, forward_main_latent=True,
             dump_forward_results=False,
             detach=False, detach_from='',
             save_raw_loss=False):
        """
        Calculate loss
        :param batch: batch of data to calculate loss
        :param expr_key: expression group to use as input (by default, 'all')
        :param forward_pheno: should calculate phenotype related losses
        :param selected_pheno: phenotype selection for loss calculation, should be None (selecting all phenotypes, and
        related losses and regularizations), or a dictionary formulated as:
        {'pheno_name': {'loss': [list of loss names] or '*' (selecting all), 'regularization': [list of regularization
        names, could be Null] or '*' or None (no regularization)}}
        :param forward_signature: should calculate signature related losses
        :param selected_signature: signature selection for loss calculation, should be None (selecting all signatures,
        and related losses and regularizations), or a dictionary formulated similar to selected_pheno
        :param forward_reconstruction: should calculate reconstruction loss (for Extractor model, when this turn on, all
         latents will be calculated by force)
        :param forward_main_latent: should calculate main latent
        :param dump_forward_results: should preserve forwarded tensors in the return dict
        :param detach: should loss be detached as specified in `detach_from`
        :param detach_from: starting from where should loss be detached
        :param save_raw_loss: when `True`, apart fro mthe weighted losses, unweighted, raw losses will also be recorded
        :return:
        """
        # Forward model (obtain pre-loss-calculated tensors)
        signature_select_fwd = list()
        if forward_signature:
            signature_select_fwd = self.select_item_dict(selection=selected_signature, internal=self.signature_config)
        pheno_select_fwd = list()
        if forward_pheno:
            pheno_select_fwd = self.select_item_dict(selection=selected_pheno, internal=self.pheno_config)
        main_fwd = forward_main_latent
        rec_fwd = forward_reconstruction
        signature_attach_req_set = set(signature_select_fwd)
        pheno_attach_req_set = set(pheno_select_fwd)
        for cur_key in pheno_select_fwd:
            model_details = self.pheno_config[cur_key].get('model')
            if model_details is not None:
                if model_details.get('attach') == 'True':
                    if model_details['attach_to'] == 'signature_lat':
                        signature_attach_req_set.add(model_details['attach_key'])
                    elif model_details['attach_to'] == 'pheno_lat':
                        pheno_attach_req_set.add(model_details['attach_key'])
                    elif model_details['attach_to'] == 'main_lat':
                        main_fwd = True
                    elif model_details['attach_to'] == 'all_lat':
                        rec_fwd = True
                    elif model_details['attach_to'] == 'multiple':
                        for cur_attach in model_details['attach_key']:
                            if cur_attach['type'] == 'pheno':
                                pheno_attach_req_set.add(cur_attach['key'])
                            elif cur_attach['type'] == 'signature':
                                signature_attach_req_set.add(cur_attach['key'])
                            elif cur_attach['type'] == 'main':
                                main_fwd = True
        for cur_key in signature_select_fwd:
            model_details = self.signature_config[cur_key].get('model')
            if model_details is not None:
                if model_details.get('attach') == 'True':
                    if model_details['attach_to'] == 'signature_lat':
                        signature_attach_req_set.add(model_details['attach_key'])
                    elif model_details['attach_to'] == 'pheno_lat':
                        pheno_attach_req_set.add(model_details['attach_key'])
                    elif model_details['attach_to'] == 'main_lat':
                        main_fwd = True
                    elif model_details['attach_to'] == 'all_lat':
                        rec_fwd = True
                    elif model_details['attach_to'] == 'multiple':
                        for cur_attach in model_details['attach_key']:
                            if cur_attach['type'] == 'pheno':
                                pheno_attach_req_set.add(cur_attach['key'])
                            elif cur_attach['type'] == 'signature':
                                signature_attach_req_set.add(cur_attach['key'])
                            elif cur_attach['type'] == 'main':
                                main_fwd = True
        signature_select_fwd = list(signature_attach_req_set)
        pheno_select_fwd = list(pheno_attach_req_set)
        if rec_fwd:
            signature_select_fwd = '*'
            pheno_select_fwd = '*'
        signature_select_fwd = self.select_item_dict(selection=signature_select_fwd, internal=self.signature_config)
        pheno_select_fwd = self.select_item_dict(selection=pheno_select_fwd, internal=self.pheno_config)

        fwd_res = self.model(batch['expr'][expr_key],
                             forward_main_latent=main_fwd,
                             forward_reconstruction=rec_fwd,
                             forward_signature=len(signature_select_fwd) > 0,
                             selected_signature=signature_select_fwd,
                             forward_pheno=len(pheno_select_fwd) > 0,
                             selected_pheno=pheno_select_fwd,
                             detach=detach, detach_from=detach_from)

        # Reconstruction Loss
        main_loss = {'loss': dict(), 'regularization': dict(),
                     'loss_raw': dict(), 'regularization_raw': dict()}
        if forward_reconstruction:
            for cur_main_loss_key in self.main_latent_config['loss'].keys():
                cur_main_loss = self.main_latent_config['loss'][cur_main_loss_key]
                if cur_main_loss['type'] == 'MSE' or cur_main_loss['type'] == 'L2':
                    main_loss['loss'][cur_main_loss_key] = torch.nn.functional.mse_loss(fwd_res['x'], fwd_res['re_x'])
                elif cur_main_loss['type'] == 'L1':
                    main_loss['loss'][cur_main_loss_key] = torch.nn.functional.l1_loss(fwd_res['x'], fwd_res['re_x'])
                elif cur_main_loss['type'] == 'Cosine':
                    # Equivalent to cosine_embedding_loss(x, y, all_one)
                    main_loss['loss'][cur_main_loss_key] = torch.nn.functional.cosine_similarity(fwd_res['x'], fwd_res['re_x'])
                    main_loss['loss'][cur_main_loss_key] = 1.0 - main_loss['loss'][cur_main_loss_key]
                    main_loss['loss'][cur_main_loss_key] = main_loss['loss'][cur_main_loss_key].mean()
                elif cur_main_loss['type'] == 'L1_norm':
                    # Calculate L1 norm penalty on reconstructed output
                    main_loss['loss'][cur_main_loss_key] = torch.linalg.norm(fwd_res['re_x'], dim=1, ord=1).mean()
                elif cur_main_loss['type'] == 'L2_norm':
                    # Calculate L2 norm penalty on reconstructed output
                    main_loss['loss'][cur_main_loss_key] = torch.linalg.norm(fwd_res['re_x'], dim=1, ord=2).mean()
                elif cur_main_loss['type'] == 'SW2':
                    # Similar to regularization, but to match the distribution of output vectors to a designated distribution
                    raise NotImplementedError('SW2 loss on output not implemented yet')
                else:
                    raise NotImplementedError("Unsupported main latent loss type")
                if save_raw_loss:
                    main_loss['loss_raw'][cur_main_loss_key] = main_loss['loss'][cur_main_loss_key]
                main_loss['loss'][cur_main_loss_key] = main_loss['loss'][cur_main_loss_key] * self.main_loss_weight['loss'][cur_main_loss_key]

        # Main latent regularizations
        if forward_main_latent:
            for cur_main_reg_loss_key in self.main_latent_config['regularization'].keys():
                cur_main_reg_loss = self.main_latent_config['regularization'][cur_main_reg_loss_key]
                if cur_main_reg_loss['type'] != 'none':
                    main_loss['regularization'][cur_main_reg_loss_key] = self.regularize(tensor=fwd_res['lat_main'],
                                                                                         regularization_config=
                                                                                         self.main_latent_config[
                                                                                             'regularization'][
                                                                                             cur_main_reg_loss_key])
                if save_raw_loss:
                    main_loss['regularization_raw'][cur_main_reg_loss_key] = main_loss['regularization'][cur_main_reg_loss_key]
                main_loss['regularization'][cur_main_reg_loss_key] = main_loss['regularization'][cur_main_reg_loss_key] * self.main_loss_weight['regularization'][cur_main_reg_loss_key]

        # Phenotype loss and regularization loss
        pheno_loss = dict()
        if forward_pheno:
            if selected_pheno is None:
                # By default, select all phenotypes and all losses included
                selected_pheno = {idx: {'loss': '*', 'regularization': '*'} for idx in self.pheno_config.keys()}
            for cur_pheno in selected_pheno.keys():
                pheno_loss[cur_pheno] = {'loss': dict(), 'regularization': dict(),
                                         'loss_raw': dict(), 'regularization_raw': dict()}

                # Phenotype loss
                selected_pheno_loss_keys = self.select_loss_dict(selection=selected_pheno[cur_pheno]['loss'],
                                                                 internal=self.pheno_config[cur_pheno]['loss'])
                for cur_pheno_loss_key in selected_pheno_loss_keys:
                    cur_pheno_loss = self.pheno_config[cur_pheno]['loss'][cur_pheno_loss_key]

                    cur_pheno_out = fwd_res['pheno_out'][cur_pheno]
                    # Handle GRL/GNL requests (re-calculate from pheno_lat)
                    if cur_pheno_loss.get('enable_GRL') == 'True':
                        cur_pheno_lat = fwd_res['lat_pheno'][cur_pheno]
                        cur_pheno_lat = ReverseLayerF.apply(cur_pheno_lat)
                        cur_pheno_out = self.model.model['pheno_models'][cur_pheno](cur_pheno_lat)
                    if cur_pheno_loss.get('enable_GNL') == 'True':
                        cur_pheno_lat = fwd_res['lat_pheno'][cur_pheno]
                        cur_pheno_lat = NeutralizeLayerF.apply(cur_pheno_lat)
                        cur_pheno_out = self.model.model['pheno_models'][cur_pheno](cur_pheno_lat)

                    if cur_pheno_loss['type'] == 'NLL':
                        # TODO: weighted NLL (to better handle label imbalance)
                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0])
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = \
                            torch.nn.functional.nll_loss(cur_pheno_out, pheno_ans)
                    elif cur_pheno_loss['type'] == 'MSE' or cur_pheno_loss['type'] == 'L2':
                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0], -1)
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = \
                            torch.nn.functional.mse_loss(cur_pheno_out, pheno_ans)
                    elif cur_pheno_loss['type'] == 'L1':
                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0], -1)
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = \
                            torch.nn.functional.l1_loss(cur_pheno_out, pheno_ans)
                    elif cur_pheno_loss['type'] == 'Cosine':
                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0], -1)
                        # Equivalent to cosine_embedding_loss(x, y, all_one)
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = torch.nn.functional.cosine_similarity(cur_pheno_out, pheno_ans)
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = 1.0 - pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key]
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key].mean()
                    elif cur_pheno_loss['type'] == 'BCE':
                        # Binary cross entropy (to do mask prediction)
                        # pheno_ans is expected to be a 1-d binary array (or [0,1] prob range)
                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0], -1)
                        if cur_pheno_loss['weighted'] == 'True':
                            # TODO: weighted BCE (to better handle label imbalance)
                            raise NotImplementedError
                        else:
                            torch.nn.functional.binary_cross_entropy(cur_pheno_out, pheno_ans)

                    else:
                        print('Unsupported phenotype supervision loss type.')
                        raise ValueError
                    if save_raw_loss:
                        pheno_loss[cur_pheno]['loss_raw'][cur_pheno_loss_key] = pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key]
                    pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] * self.pheno_loss_weight[cur_pheno]['loss'][cur_pheno_loss_key]

                # Phenotype regularization loss
                selected_pheno_reg_loss_keys = self.select_loss_dict(
                    selection=selected_pheno[cur_pheno]['regularization'],
                    internal=self.pheno_config[cur_pheno]['regularization'])
                for cur_pheno_reg_loss_key in selected_pheno_reg_loss_keys:
                    cur_pheno_reg_loss = self.pheno_config[cur_pheno]['regularization'][cur_pheno_reg_loss_key]
                    if cur_pheno_reg_loss['type'] != 'none':
                        cur_pheno_lat = fwd_res['lat_pheno'][cur_pheno]
                        # Handle GRL/GNL requests
                        if cur_pheno_reg_loss.get('enable_GRL') == 'True':
                            cur_pheno_lat = ReverseLayerF.apply(cur_pheno_lat)
                        if cur_pheno_reg_loss.get('enable_GNL') == 'True':
                            cur_pheno_lat = NeutralizeLayerF.apply(cur_pheno_lat)

                        pheno_ans = batch['pheno'][cur_pheno].squeeze().reshape(fwd_res['pheno_out'][cur_pheno].shape[0], -1)
                        pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key] = \
                            self.regularize(tensor=cur_pheno_lat,
                                            regularization_config=self.pheno_config[cur_pheno]['regularization'][
                                                cur_pheno_reg_loss_key],
                                            supervision=pheno_ans)
                        if save_raw_loss:
                            pheno_loss[cur_pheno]['regularization_raw'][cur_pheno_reg_loss_key] = pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key]
                        pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key] = pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key] * \
                                                                                          self.pheno_loss_weight[cur_pheno]['regularization'][cur_pheno_reg_loss_key]

        # Signature loss and regularization
        signature_loss = dict()
        if forward_signature:
            if selected_signature is None:
                # Select all signature
                selected_signature = {idx: {'loss': '*', 'regularization': '*'} for idx in self.signature_config.keys()}
            for cur_signature in selected_signature.keys():
                signature_loss[cur_signature] = {"loss": dict(), "regularization": dict(),
                                                 "loss_raw": dict(), "regularization_raw": dict()}
                ## Signature loss
                selected_signature_loss_keys = self.select_loss_dict(
                    selection=selected_signature[cur_signature]['loss'],
                    internal=self.signature_config[cur_signature]['loss'])
                for cur_signature_loss_key in selected_signature_loss_keys:
                    cur_signature_loss = self.signature_config[cur_signature]['loss'][cur_signature_loss_key]

                    cur_signature_out = fwd_res['signature_out'][cur_signature]
                    # Handle GRL/GNL requests (re-calc)
                    if cur_signature_loss.get('enable_GRL') == 'True':
                        cur_signature_lat = fwd_res['lat_signature'][cur_signature]
                        cur_signature_lat = ReverseLayerF.apply(cur_signature_lat)
                        cur_signature_out = self.model.model['signature_regressors'][cur_signature](cur_signature_lat)
                    if cur_signature_loss.get('enable_GNL') == 'True':
                        cur_signature_lat = fwd_res['lat_signature'][cur_signature]
                        cur_signature_lat = NeutralizeLayerF.apply(cur_signature_lat)
                        cur_signature_out = self.model.model['signature_regressors'][cur_signature](cur_signature_lat)

                    signature_ans = batch['expr'][cur_signature].squeeze().reshape(cur_signature_out.shape[0], -1)

                    if cur_signature_loss['type'] == 'MSE' or cur_signature_loss['type'] == 'L2':
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = \
                            torch.nn.functional.mse_loss(cur_signature_out, signature_ans)
                    elif cur_signature_loss['type'] == 'L1':
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = \
                            torch.nn.functional.l1_loss(cur_signature_out, signature_ans)
                    elif cur_signature_loss['type'] == 'Cosine':
                        # Equivalent to cosine_embedding_loss(x, y, all_one)
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = \
                            torch.nn.functional.cosine_similarity(cur_signature_out, signature_ans)
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = 1.0 - signature_loss[cur_signature]['loss'][cur_signature_loss_key]
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = signature_loss[cur_signature]['loss'][cur_signature_loss_key].mean()
                    elif cur_signature_loss['type'] == 'SW2':
                        # TODO: SW2 loss to align distrubution of signature output to prior distribution
                        raise NotImplementedError
                    elif cur_signature_loss['type'] == 'BCE':
                        # TODO: Weighted BCE loss
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = torch.nn.functional.binary_cross_entropy(cur_signature_out, signature_ans)
                        print("Debugging BCE")
                        print("cur_signature_out", cur_signature_out)
                        print("cur_signature_out.shape", cur_signature_out.shape)
                        print("signature_ans", signature_ans)
                        print("signature_ans.shape", signature_ans.shape)
                        print("signature_loss[cur_signature]['loss'][cur_signature_loss_key]", signature_loss[cur_signature]['loss'][cur_signature_loss_key])
                    else:
                        print('Unsupported signature supervision loss type.')
                        raise ValueError
                    if save_raw_loss:
                        signature_loss[cur_signature]['loss_raw'][cur_signature_loss_key] = signature_loss[cur_signature]['loss'][cur_signature_loss_key]
                    signature_loss[cur_signature]['loss'][cur_signature_loss_key] = signature_loss[cur_signature]['loss'][cur_signature_loss_key] * self.signature_loss_weight[cur_signature]['loss'][
                        cur_signature_loss_key]

                ## Signature latent regularization
                selected_signature_reg_loss_keys = \
                    self.select_loss_dict(selection=selected_signature[cur_signature]['regularization'],
                                          internal=self.signature_config[cur_signature]['regularization'])
                for cur_signature_reg_loss_key in selected_signature_reg_loss_keys:
                    cur_signature_reg_loss = self.signature_config[cur_signature]['regularization'][
                        cur_signature_reg_loss_key]
                    if cur_signature_reg_loss['type'] != 'none':
                        # TODO: support of supervised regularization (e.g. approximate regions based on bins of expression values)

                        # Handle GRL/GNL requests
                        cur_signature_lat = fwd_res['lat_signature'][cur_signature]
                        if cur_signature_reg_loss.get('enable_GRL') == 'True':
                            cur_signature_lat = ReverseLayerF.apply(cur_signature_lat)
                        if cur_signature_reg_loss.get('enable_GNL') == 'True':
                            cur_signature_lat = NeutralizeLayerF.apply(cur_signature_lat)

                        signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key] = \
                            self.regularize(tensor=cur_signature_lat,
                                            regularization_config=
                                            self.signature_config[cur_signature]['regularization'][
                                                cur_signature_reg_loss_key])
                        if save_raw_loss:
                            signature_loss[cur_signature]['regularization_raw'][cur_signature_reg_loss_key] = signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key]
                        signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key] = signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key] * \
                                                                                                      self.signature_loss_weight[cur_signature]['regularization'][cur_signature_reg_loss_key]

        ret = {
            'main_latent_loss': main_loss,
            'pheno_loss': pheno_loss,
            'signature_loss': signature_loss
        }
        if dump_forward_results:
            ret['fwd_res'] = fwd_res

        return ret

    def train(self, batch,
              backward_reconstruction_loss=True, backward_main_latent_regularization=True,
              backward_pheno_loss=True, selected_pheno: dict = None,
              backward_signature_loss=True, selected_signature: dict = None,
              suppress_backward=False,
              detach=False, detach_from='', save_raw_loss=False):
        """
        Train model using specified batch.
        :param batch: batched data, obtained from rna_count dataset
        :param backward_reconstruction_loss: should optimize/backward reconstruction loss, default True
        :param backward_main_latent_regularization: should optimize/backward regularization of main latent space, default True
        :param backward_pheno_loss: should optimize/backward phenotype-related loss, default True
        :param selected_pheno: a list of selected phenotype to backward, default None; when None, all phenotypes set when initializing the Controller object will be used
        :param backward_signature_loss: should optimize/backward signature-related loss, default True
        :param selected_signature: a list of selected signatures to backward, default None; when None, all signatures set when initializing the Controller object will be selected
        :param suppress_backward: should backward of sum of losses be suppressed (useful when external training agent override the control)
        :return: A dict, containing losses used in this training tick
        """
        # Switch model to train mode
        self.model.train()

        # Forward model
        loss = self.loss(batch,
                         expr_key='all',
                         forward_reconstruction=backward_reconstruction_loss,
                         forward_main_latent=backward_main_latent_regularization,
                         forward_pheno=backward_pheno_loss, selected_pheno=selected_pheno,
                         forward_signature=backward_signature_loss, selected_signature=selected_signature,
                         dump_forward_results=False,
                         detach=detach, detach_from=detach_from, save_raw_loss=save_raw_loss)

        total_loss = torch.Tensor([0.])
        if self.device == 'cuda':
            total_loss.cuda()

        # Reconstruction loss
        if backward_reconstruction_loss:
            for cur_main_loss in loss['main_latent_loss']['loss'].values():
                total_loss += cur_main_loss

        # Main latent regularization
        if backward_main_latent_regularization:
            for cur_main_reg_loss in loss['main_latent_loss']['regularization'].values():
                total_loss += cur_main_reg_loss

        # Phenotype loss
        if backward_pheno_loss:
            for cur_group in ['loss', 'regularization']:
                for cur_pheno in loss['pheno_loss'].keys():
                    for cur_pheno_loss in loss['pheno_loss'][cur_pheno][cur_group].values():
                        total_loss += cur_pheno_loss

        # Signature loss
        if backward_signature_loss:
            for cur_group in ['loss', 'regularization']:
                for cur_signature in loss['signature_loss'].keys():
                    for cur_signature_loss in loss['signature_loss'][cur_signature][cur_group].values():
                        total_loss += cur_signature_loss

        loss['total_loss_backwarded'] = total_loss

        # Verbose output for debug
        if self.verbose:
            print(loss)

        if suppress_backward:
            # Directly return loss dict when backward is suppressed
            return loss

        # Optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss

    def eval(self, batch,
             forward_pheno=False, selected_pheno=None,
             forward_signature=False, selected_signature=None,
             forward_reconstruction=False, forward_main_latent=False, dump_latent=False, save_raw_loss=False):
        """
        Evaluate losses.
        :param batch:
        :param forward_pheno:
        :param selected_pheno:
        :param forward_signature:
        :param selected_signature:
        :param dump_latent:
        :return:
        """
        # Switch model to eval mode
        self.model.eval()

        loss = None
        with torch.no_grad():
            loss = self.loss(batch=batch,
                             expr_key='all', forward_main_latent=forward_main_latent,
                             forward_pheno=forward_pheno, selected_pheno=selected_pheno,
                             forward_signature=forward_signature, selected_signature=selected_signature,
                             forward_reconstruction=forward_reconstruction, dump_forward_results=dump_latent, save_raw_loss=save_raw_loss)

        if self.verbose:
            torch.set_printoptions(threshold=1, edgeitems=1, profile='short')
            print(loss)

        return loss

    def save_checkpoint(self, save_config=False, save_model_arch=False):
        ret_state_dict = dict()

        ret_state_dict['type'] = 'extractor'

        # Configs (redundant for checking)
        if save_config:
            ret_state_dict['config'] = self.config
            ret_state_dict['main_latent_config'] = self.main_latent_config
            ret_state_dict['pheno_config'] = self.pheno_config
            ret_state_dict['signature_config'] = self.signature_config

        # Epochs
        ret_state_dict['cur_epoch'] = self.cur_epoch
        ret_state_dict['cur_tick'] = self.cur_tick
        ret_state_dict['main_epoch'] = self.main_epoch
        ret_state_dict['pheno_epoch'] = self.pheno_epoch
        ret_state_dict['signature_epoch'] = self.signature_epoch

        # Loss weights
        ret_state_dict['main_loss_weight'] = self.main_loss_weight
        ret_state_dict['pheno_loss_weight'] = self.pheno_loss_weight
        ret_state_dict['signature_loss_weight'] = self.signature_loss_weight

        # Model state dict
        ret_state_dict['model_state_dict'] = self.model.state_dict()

        # Optimizer state dict
        ret_state_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        # Model architecture
        if save_model_arch:
            ret_state_dict['model'] = self.model

        return ret_state_dict

    def load_checkpoint(self, state_dict: dict):
        if type(state_dict) is not dict:
            raise ValueError('A dict generated by Extractor controllor\'s save_checkpoint is expected.')
        if state_dict.get('type') != 'extractor':
            raise ValueError('Input state dict is not for Extractor.')

        # Verbose logging
        if self.verbose:
            print("Controller to resume state:")
            print(state_dict)

        # Resume Epochs
        self.cur_epoch = state_dict['cur_epoch']
        self.cur_tick = state_dict['cur_tick']
        self.main_epoch = state_dict['main_epoch']
        self.pheno_epoch = state_dict['pheno_epoch']
        self.signature_epoch = state_dict['signature_epoch']

        # Resume loss weights
        self.main_loss_weight = state_dict['main_loss_weight']
        self.pheno_loss_weight = state_dict['pheno_loss_weight']
        self.signature_loss_weight = state_dict['signature_loss_weight']

        # Resume model state dict
        self.model.load_state_dict(state_dict['model_state_dict'])

        # Resume optimizer state dict
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
