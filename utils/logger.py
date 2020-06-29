import json
import pickle
from pathlib import Path

import pandas as pd
import tensorboardX


class Logger(object):
    def __init__(self, log_path='./logs/'):

        # Arguments
        self.log_path = log_path

        # Create logging directory if not exist
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # SummaryWriter from tensorboardX
        self.log_writer = tensorboardX.SummaryWriter(self.log_path)

    def log_loss(self, trainer_output:dict, tick,
                 loss_name_prefix=''):

        # TODO: Selectively log losses

        # Class label supervision loss and regularization loss
        for cur_pheno in trainer_output['pheno_loss'].keys():
            for cur_group in ['loss', 'regularization']:
                for cur_loss_key in trainer_output['pheno_loss'][cur_pheno][cur_group].keys():
                    self.log_writer.add_scalar(loss_name_prefix + '_pheno_' + cur_pheno + '_' + cur_group + '_' + cur_loss_key,
                                               trainer_output['pheno_loss'][cur_pheno][cur_group][cur_loss_key],
                                               tick)


        # Signature supervision loss
        for cur_signature in trainer_output['signature_loss'].keys():
            for cur_group in ['loss', 'regularization']:
                for cur_loss_key in trainer_output['signature_loss'][cur_signature][cur_group].keys():
                    self.log_writer.add_scalar(
                        loss_name_prefix + '_signature_' + cur_signature + '_' + cur_group + '_' + cur_loss_key,
                        trainer_output['signature_loss'][cur_signature][cur_group][cur_loss_key],
                        tick)


        # Main latent loss
        for cur_group in ['loss', 'regularization']:
            for cur_loss_key in trainer_output['main_latent_loss'][cur_group].keys():
                self.log_writer.add_scalar(loss_name_prefix + '_main_latent_' + cur_group + '_' + cur_loss_key,
                                           trainer_output['main_latent_loss'][cur_group][cur_loss_key],
                                           tick)

    def dump_latent_to_csv(self, controller_output,
                           dump_main=True,
                           dump_pheno=True, selected_pheno=None,
                           dump_signature=True, selected_signature=None,
                           rownames=None,
                           path='latent.csv'):
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
        # Main latent
        if dump_main:
            lat_all.append(
                pd.DataFrame(controller_output['fwd_res']['lat_main'].numpy()).set_axis(['main.' + str(i + 1) for i in range(
                    controller_output['fwd_res']['lat_main'].shape[1]
                )], axis=1, inplace=False)
            )

        # Class label latents
        if dump_pheno:
            if selected_pheno is None:
                raise ValueError
            for cur_pheno in selected_pheno:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_pheno'][cur_pheno].numpy()).set_axis(
                        [cur_pheno + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_pheno'][cur_pheno].shape[1]
                        )], axis=1, inplace=False)
                )

        # Signature latents
        if dump_signature:
            if selected_signature is None:
                raise ValueError
            for cur_signature in selected_signature:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_signature'][cur_signature].numpy()).set_axis(
                        [cur_signature + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_signature'][cur_signature].shape[1]
                        )], axis=1, inplace=False)
                )

        # Assemble dataframe
        if rownames is not None:
            lat_all = pd.concat(lat_all, axis=1).set_index(rownames)

        # Save CSV
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