import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import tensorboardX


class Logger(object):
    def __init__(self, log_path='./logs/', suppress_tensorboardX=False):

        # Arguments
        self.log_path = log_path

        # Create logging directory if not exist
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # SummaryWriter from tensorboardX
        self.log_writer = None
        if not suppress_tensorboardX:
            self.log_writer = tensorboardX.SummaryWriter(self.log_path)



    def log_loss(self, trainer_output: dict, tick,
                 loss_name_prefix='',
                 selected_loss_group=['loss', 'regularization']):

        # TODO: Selectively log losses

        # Class label supervision loss and regularization loss
        for cur_pheno in trainer_output['pheno_loss'].keys():
            for cur_group in selected_loss_group:
                for cur_loss_key in trainer_output['pheno_loss'][cur_pheno][cur_group].keys():
                    self.log_writer.add_scalar(loss_name_prefix + '_pheno_' + cur_pheno + '_' + cur_group + '_' + cur_loss_key,
                                               trainer_output['pheno_loss'][cur_pheno][cur_group][cur_loss_key],
                                               tick)


        # Signature supervision loss
        for cur_signature in trainer_output['signature_loss'].keys():
            for cur_group in selected_loss_group:
                for cur_loss_key in trainer_output['signature_loss'][cur_signature][cur_group].keys():
                    self.log_writer.add_scalar(
                        loss_name_prefix + '_signature_' + cur_signature + '_' + cur_group + '_' + cur_loss_key,
                        trainer_output['signature_loss'][cur_signature][cur_group][cur_loss_key],
                        tick)

        # Main latent loss
        for cur_group in selected_loss_group:
            for cur_loss_key in trainer_output['main_latent_loss'][cur_group].keys():
                self.log_writer.add_scalar(loss_name_prefix + '_main_latent_' + cur_group + '_' + cur_loss_key,
                                           trainer_output['main_latent_loss'][cur_group][cur_loss_key],
                                           tick)

    def log_parameter(self, trainer_output: dict, tick,
                      log_prefix=''):
        # TODO: log parameters on Tensorboard
        raise NotImplementedError

    def log_metric(self, trainer_output: dict, tick, metric_configs: dict,
                   log_prefix=''):
        # TODO: log metrics (AUROC, AUPR, etc.) on Tensorboard
        for cur_metric in metric_configs:
            if cur_metric['type'] == 'AUROC':
                raise NotImplementedError
            elif cur_metric['type'] == 'AUPR':
                # Log AUPR only
                raise NotImplementedError
            elif cur_metric['type'] == 'PR':
                # Log P, R, AUPR, and PR Curve
                raise NotImplementedError
            elif cur_metric['type'] == 'ROC':
                # Log ROC Curve
                raise NotImplementedError
            else:
                raise NotImplementedError

    def dump_latent_to_csv(self, controller_output,
                           dump_main=True,
                           dump_lat_pre=False,
                           dump_pheno=True, dump_pheno_out=False, selected_pheno=None,
                           dump_signature=True, dump_signature_out=False, selected_signature=None,
                           dump_re_x=False, re_x_col_naming='dimid',
                           rownames=None, colnames=None,
                           path='latent.csv',
                           compression='none'):
        """
        Dump latent code from the output of the controller.
        :param trainer_output: (dict) a dictionary returned by the controller (using eval_all method, with dump_latent=True)
        :param dump_main: (bool) should dump main latent space
        :param dump_pheno: (bool) should dump phenotype latent space
        :param selected_pheno: (list of str) a list containing phenotype ids to be dumped
        :param dump_signature: (bool) should dump signature latent space
        :param selected_signature: (list of str) a list containing signature ids to be dumped
        :param path: (str) path for the CSV file storing
        :return:
        """

        lat_all = list()
        # Main latent (lat_main)
        if dump_main:
            lat_all.append(
                pd.DataFrame(controller_output['fwd_res']['lat_main'].numpy()).set_axis(['main.' + str(i + 1) for i in range(
                    controller_output['fwd_res']['lat_main'].shape[1]
                )], axis=1, inplace=False)
            )

        # Reconstructed input
        # Matrix to dump may be very large
        if dump_re_x:
            if re_x_col_naming == 'dimid':
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['re_x'].numpy()).set_axis(['re_x.' + str(i + 1) for i in range(
                        controller_output['fwd_res']['re_x'].shape[1]
                    )], axis=1, inplace=False)
                )
            elif re_x_col_naming == 'genenames':
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['re_x'].numpy()).set_axis(colnames, axis=1, inplace=False)
                )

        # Pre-encoder results (lat_pre)
        if dump_lat_pre:
            lat_all.append(
                pd.DataFrame(controller_output['fwd_res']['lat_pre'].numpy()).set_axis(['lat_pre.' + str(i + 1) for i in range(
                    controller_output['fwd_res']['lat_pre'].shape[1]
                )], axis=1, inplace=False)
            )

        # Phenotype latents
        if dump_pheno:
            if selected_pheno is None:
                selected_pheno = controller_output['fwd_res']['lat_pheno'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_pheno in selected_pheno:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_pheno'][cur_pheno].numpy()).set_axis(
                        [cur_pheno + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_pheno'][cur_pheno].shape[1]
                        )], axis=1, inplace=False)
                )

        # Phenotype outputs
        if dump_pheno_out:
            if selected_pheno is None:
                selected_pheno = controller_output['fwd_res']['pheno_out'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_pheno in selected_pheno:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['pheno_out'][cur_pheno].numpy()).set_axis(
                        [cur_pheno + '.out.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['pheno_out'][cur_pheno].shape[1]
                        )], axis=1, inplace=False)
                )

        # Signature latents
        if dump_signature:
            if selected_signature is None:
                selected_pheno = controller_output['fwd_res']['lat_signature'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_signature in selected_signature:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_signature'][cur_signature].numpy()).set_axis(
                        [cur_signature + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_signature'][cur_signature].shape[1]
                        )], axis=1, inplace=False)
                )

        # Signature outputs
        if dump_signature_out:
            if selected_signature is None:
                selected_pheno = controller_output['fwd_res']['signature_out'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_signature in selected_signature:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['signature_out'][cur_signature].numpy()).set_axis(
                        [cur_signature + '.out.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['signature_out'][cur_signature].shape[1]
                        )], axis=1, inplace=False)
                )

        # Assemble dataframe (and set cell names)
        if rownames is not None:
            lat_all = pd.concat(lat_all, axis=1).set_index(rownames)

        # Save CSV
        if compression == 'hdf':
            lat_all.to_hdf(path, 'lat_all', mode='w')
        elif compression != 'none':
            lat_all.to_csv(path, compression=compression)
        else:
            lat_all.to_csv(path)

    def save_config(self, config_dict, json_path):
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)

    def save_splits(self, split_dict, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(split_dict, f)

    def checkpoint(self, model, trainer,
                   path='./checkpoints/latest/'):
        # TODO: save model state for checkpointing
        raise NotImplementedError