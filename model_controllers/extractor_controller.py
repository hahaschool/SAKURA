import functools

import torch
import torch.optim

import utils.distributions as distributions
from models.extractor import Extractor
from utils.sliced_wasserstein import SlicedWasserstein


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
        if self.device is 'cuda':
            self.model.cuda()

        # Setup Optimizer
        if self.config['optimizer']['type'] == 'RMSProp':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                 lr=self.config['optimizer']['RMSProp_lr'],
                                                 alpha=self.config['optimizer']['RMSProp_alpha'])
        else:
            print("Optimizers other than RMSProp not implemented.")
            raise NotImplementedError

        if verbose:
            print('===========================')
            print('Extractor controller')
            print(self.pheno_config)
            print(self.signature_config)
            print(self.main_latent_config)
            print(self.optimizer)
            print('===========================')

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

        # Main latent epoch weight progression
        if prog_main:
            # TODO: also implement selection of main latent loss/regularization here (not sure if useful)
            for cur_group in ['loss', 'regularization']:
                for cur_main_loss_key in self.main_latent_config[cur_group].keys():
                    self.main_epoch[cur_group][cur_main_loss_key] += 1
                    if self.main_epoch[cur_group][cur_main_loss_key] >= \
                            self.main_latent_config[cur_group][cur_main_loss_key]['progressive_start_epoch']:
                        self.main_loss_weight[cur_group][cur_main_loss_key] *= \
                            self.main_latent_config[cur_group][cur_main_loss_key]['progressive_const']

        # Phenotype supervision and regularization loss weight progression
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
                            self.pheno_loss_weight[cur_pheno][cur_group][cur_pheno_loss_key] *= \
                                self.pheno_config[cur_pheno][cur_group][cur_pheno_loss_key]['progressive_const']

        # Signature supervision and regularization loss progression
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
                            self.signature_loss_weight[cur_signature][cur_group][cur_signature_loss_key] *= \
                                self.signature_config[cur_signature][cur_group][cur_signature_loss_key][
                                    'progressive_const']

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
        elif regularization_config['type'] == 'SW2_gaussian_mixturn_supervised':
            # TODO: Supervised gaussian mixture prior
            raise NotImplementedError
        elif regularization_config['type'] == 'euclidean_anchor':
            # TODO: Euclidean anchoring
            raise NotImplementedError

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
            if selection is None or selection == '*':
                # Select all in internal config
                return internal.keys()
            elif type(selection) is dict:
                return selection.keys()
            elif type(selection) is list:
                return selection
            elif type(selection) is str:
                return [selection]
            else:
                raise ValueError
        elif mode == 'dict':
            if selection is None or selection == '*':
                return {idx: '*' for idx in internal.keys()}
            elif type(selection) is dict():
                return selection

    def loss(self, batch, expr_key='all',
             forward_pheno=True, selected_pheno=None,
             forward_signature=True, selected_signature=None,
             forward_reconstruction=True, forward_main_latent=True,
             dump_forward_results=False):
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
        :return:
        """
        # Forward model (obtain pre-loss-calculated tensors)
        fwd_res = self.model(batch['expr'][expr_key],
                             forward_main_latent=forward_main_latent,
                             forward_reconstruction=forward_reconstruction,
                             forward_signature=forward_signature,
                             selected_signature=self.select_item_dict(selection=selected_signature,
                                                                      internal=self.signature_config),
                             forward_pheno=forward_pheno,
                             selected_pheno=self.select_item_dict(selection=selected_pheno,
                                                                  internal=self.pheno_config))

        # Reconstruction Loss
        main_loss = {'loss': dict(), 'regularization': dict()}
        if forward_reconstruction:
            for cur_main_loss_key in self.main_latent_config['loss'].keys():
                cur_main_loss = self.main_latent_config['loss'][cur_main_loss_key]
                if cur_main_loss['type'] == 'MSE' or cur_main_loss['type'] == 'L2':
                    main_loss['loss'][cur_main_loss_key] = torch.nn.functional.mse_loss(fwd_res['x'], fwd_res['re_x'])
                elif cur_main_loss['type'] == 'L1':
                    main_loss['loss'][cur_main_loss_key] = torch.nn.functional.l1_loss(fwd_res['x'], fwd_res['re_x'])
                else:
                    print("Unsupported main latent loss type")
                    raise NotImplementedError
                main_loss['loss'][cur_main_loss_key] *= self.main_loss_weight['loss'][cur_main_loss_key]

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
                main_loss['regularization'][cur_main_reg_loss_key] *= self.main_loss_weight['regularization'][
                    cur_main_reg_loss_key]

        # Phenotype loss and regularization loss
        pheno_loss = dict()
        if forward_pheno:
            if selected_pheno is None:
                # By default, select all phenotypes and all losses included
                selected_pheno = {idx: {'loss': '*', 'regularization': '*'} for idx in self.pheno_config.keys()}
            for cur_pheno in selected_pheno.keys():
                pheno_loss[cur_pheno] = {'loss': dict(), 'regularization': dict()}
                pheno_ans = batch['pheno'][cur_pheno].squeeze()
                # Phenotype loss
                selected_pheno_loss_keys = self.select_loss_dict(selection=selected_pheno[cur_pheno]['loss'],
                                                                 internal=self.pheno_config[cur_pheno]['loss'])
                for cur_pheno_loss_key in selected_pheno_loss_keys:
                    cur_pheno_loss = self.pheno_config[cur_pheno]['loss'][cur_pheno_loss_key]
                    if cur_pheno_loss['type'] == 'NLL':
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = \
                            torch.nn.functional.nll_loss(fwd_res['pheno_out'][cur_pheno], pheno_ans)
                    elif cur_pheno_loss['type'] == 'MSE' or cur_pheno_loss['type'] == 'L2':
                        pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] = \
                            torch.nn.functional.mse_loss(fwd_res['pheno_out'][cur_pheno], pheno_ans)
                    else:
                        print('Unsupported phenotype supervision loss type.')
                        raise ValueError
                    pheno_loss[cur_pheno]['loss'][cur_pheno_loss_key] *= self.pheno_loss_weight[cur_pheno]['loss'][
                        cur_pheno_loss_key]
                # Phenotype regularization loss
                selected_pheno_reg_loss_keys = self.select_loss_dict(
                    selection=selected_pheno[cur_pheno]['regularization'],
                    internal=self.pheno_config[cur_pheno]['regularization'])
                for cur_pheno_reg_loss_key in selected_pheno_reg_loss_keys:
                    cur_pheno_reg_loss = self.pheno_config[cur_pheno]['regularization'][cur_pheno_reg_loss_key]
                    if cur_pheno_reg_loss['type'] != 'none':
                        pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key] = \
                            self.regularize(tensor=fwd_res['lat_pheno'][cur_pheno],
                                            regularization_config=self.pheno_config[cur_pheno]['regularization'][
                                                cur_pheno_reg_loss_key],
                                            supervision=pheno_ans)
                        pheno_loss[cur_pheno]['regularization'][cur_pheno_reg_loss_key] *= \
                            self.pheno_loss_weight[cur_pheno]['regularization'][cur_pheno_reg_loss_key]

        # Signature loss and regularization
        signature_loss = dict()
        if forward_signature:
            if selected_signature is None:
                # Select all signature
                selected_signature = {idx: {'loss': '*', 'regularization': '*'} for idx in self.signature_config.keys()}
            for cur_signature in selected_signature.keys():
                signature_loss[cur_signature] = {"loss": dict(), "regularization": dict()}
                signature_ans = batch['expr'][cur_signature].squeeze()
                # Signature loss
                selected_signature_loss_keys = self.select_loss_dict(
                    selection=selected_signature[cur_signature]['loss'],
                    internal=self.signature_config[cur_signature]['loss'])
                for cur_signature_loss_key in selected_signature_loss_keys:
                    cur_signature_loss = self.signature_config[cur_signature]['loss'][cur_signature_loss_key]
                    if cur_signature_loss['type'] == 'MSE' or cur_signature_loss['type'] == 'L2':
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = \
                            torch.nn.functional.mse_loss(fwd_res['signature_out'][cur_signature], signature_ans)
                    elif cur_signature_loss['type'] == 'L1':
                        signature_loss[cur_signature]['loss'][cur_signature_loss_key] = \
                            torch.nn.functional.l1_loss(fwd_res['signature_out'][cur_signature], signature_ans)
                    else:
                        print('Unsupported signature supervision loss type.')
                        raise ValueError
                    signature_loss[cur_signature]['loss'][cur_signature_loss_key] *= \
                        self.signature_loss_weight[cur_signature]['loss'][cur_signature_loss_key]
                # Signature latent regularization
                selected_signature_reg_loss_keys = \
                    self.select_loss_dict(selection=selected_signature[cur_signature]['regularization'],
                                          internal=self.signature_config[cur_signature]['regularization'])
                for cur_signature_reg_loss_key in selected_signature_reg_loss_keys:
                    cur_signature_reg_loss = self.signature_config[cur_signature]['regularization'][
                        cur_signature_reg_loss_key]
                    if cur_signature_reg_loss['type'] != 'none':
                        # TODO: support of supervised regularization (e.g. approximate regions based on bins of expression values)
                        signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key] = \
                            self.regularize(tensor=fwd_res['lat_signature'][cur_signature],
                                            regularization_config=
                                            self.signature_config[cur_signature]['regularization'][
                                                cur_signature_reg_loss_key])
                        signature_loss[cur_signature]['regularization'][cur_signature_reg_loss_key] *= \
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
              backward_signature_loss=True, selected_signature: dict = None):
        """
        Train model using specified batch.
        :param batch: batched data, obtained from rna_count dataset
        :param backward_reconstruction_loss: should optimize/backward reconstruction loss, default True
        :param backward_main_latent_regularization: should optimize/backward regularization of main latent space, default True
        :param backward_pheno_loss: should optimize/backward phenotype-related loss, default True
        :param selected_pheno: a list of selected phenotype to backward, default None; when None, all phenotypes set when initializing the Controller object will be used
        :param backward_signature_loss: should optimize/backward signature-related loss, default True
        :param selected_signature: a list of selected signatures to backward, default None; when None, all signatures set when initializing the Controller object will be selected
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
                         dump_forward_results=False)

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

        # Optimizer
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Verbose output for debug
        if self.verbose:
            print(loss)

        return loss

    def eval(self, batch,
             forward_pheno=False, selected_pheno=None,
             forward_signature=False, selected_signature=None,
             forward_reconstruction=False, forward_main_latent=False, dump_latent=False):
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
                             forward_reconstruction=forward_reconstruction, dump_forward_results=dump_latent)

        if self.verbose:
            torch.set_printoptions(threshold=1, edgeitems=1, profile='short')
            print(loss)

        return loss

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
