import torch

import models.modules as model


class Extractor(torch.nn.Module):
    """
    Create extra dimensions correspoding to external information, along with main embeddings.
    Each group of additional information will create new dimensions (latent spaces).

    """

    def __init__(self,
                 input_dim : int,
                 signature_config=None, pheno_config=None,
                 encoder_neurons=50, decoder_neurons=50, main_latent_dim=2,
                 verbose=False):
        super(Extractor, self).__init__()

        # Verbose logging for debugging
        self.verbose = verbose

        self.input_dim = input_dim
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.main_latent_dim = main_latent_dim
        self.signature_config = signature_config
        self.pheno_config = pheno_config

        # Pre-encoder
        pre_encoder = model.FCPreEncoder(input_dim=self.input_dim,
                                         output_dim=self.encoder_neurons,
                                         hidden_neurons=self.encoder_neurons)

        # Main Latent Compressor
        main_latent_compressor = model.FCCompressor(input_dim=self.encoder_neurons,
                                                    output_dim=self.main_latent_dim)

        # Signature Latent Compressor
        total_latent_dim = self.main_latent_dim
        signature_latent_compressors = torch.nn.ModuleDict()
        if self.signature_config is not None:
            for cur_signature in self.signature_config.keys():
                model_details = self.signature_config[cur_signature].get('model')
                if model_details is None or model_details.get('attach') != 'True':
                    # Create extra dimensions by default
                    total_latent_dim = total_latent_dim + self.signature_config[cur_signature]['signature_lat_dim']
                    signature_latent_compressors[cur_signature] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                                     output_dim=
                                                                                     self.signature_config[cur_signature][
                                                                                         'signature_lat_dim'])

        # Signature regressor
        signature_regressors = torch.nn.ModuleDict()
        if self.signature_config is not None:
            for cur_signature in self.signature_config.keys():
                model_details = self.signature_config[cur_signature].get('model')
                if model_details is None:
                    # Legacy: by default, use a linear regressor
                    signature_regressors[cur_signature] = model.LinRegressor(
                        input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                        output_dim=self.signature_config[cur_signature]['signature_out_dim'])
                else:
                    if model_details['type'] == 'FCRegressor':
                        signature_regressors[cur_signature] = model.FCRegressor(
                            input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                            output_dim=self.signature_config[cur_signature]['signature_out_dim'],
                            hidden_neurons=model_details.get('hidden_neurons', 5)
                        )
                    elif model_details['type'] == 'LinRegressor':
                        signature_regressors[cur_signature] = model.LinRegressor(
                            input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                            output_dim=self.signature_config[cur_signature]['signature_out_dim']
                        )
                    elif model_details['type'] == 'bypass' or model_details['type'] == 'Identity':
                        signature_regressors[cur_signature] = torch.nn.Identity()
                    else:
                        raise ValueError('Unsupported phenotype side task model')

        # Phenotype Latent Compressor
        pheno_latent_compressors = torch.nn.ModuleDict()
        if self.pheno_config is not None:
            for cur_pheno in self.pheno_config.keys():
                model_details = self.pheno_config[cur_pheno].get('model')
                if model_details is None or model_details.get('attach') != 'True':
                    total_latent_dim = total_latent_dim + self.pheno_config[cur_pheno]['pheno_lat_dim']
                    pheno_latent_compressors[cur_pheno] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                             output_dim=
                                                                             self.pheno_config[cur_pheno]['pheno_lat_dim'])

        # Category classifier(/regressor)
        pheno_models = torch.nn.ModuleDict()
        if self.pheno_config is not None:
            for cur_pheno in self.pheno_config.keys():
                model_details = self.pheno_config[cur_pheno].get('model')
                if model_details is None:
                    # Legacy: by default, use a linear classifier
                    pheno_models[cur_pheno] = model.LinClassifier(
                        input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                        output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                    )
                else:
                    if model_details['type'] == "LinClassifier":
                        pheno_models[cur_pheno] = model.LinClassifier(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                        )
                    elif model_details['type'] == 'FCClassifier':
                        pheno_models[cur_pheno] = model.FCClassifier(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                        )
                    elif model_details['type'] == 'FCRegressor':
                        pheno_models[cur_pheno] = model.FCRegressor(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim'],
                            hidden_neurons=model_details.get('hidden_neurons', 5)
                        )
                    elif model_details['type'] == 'LinRegressor':
                        pheno_models[cur_pheno] = model.LinRegressor(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                        )
                    elif model_details['type'] == 'bypass' or model_details['type'] == 'Identity':
                        pheno_models[cur_pheno] = torch.nn.Identity()
                    else:
                        raise ValueError('Unsupported phenotype side task model')

        # Decoder
        decoder = model.FCDecoder(input_dim=total_latent_dim,
                                  output_dim=self.input_dim,
                                  hidden_neurons=self.decoder_neurons)

        # Assemble model
        self.model = torch.nn.ModuleDict({
            'pre_encoder': pre_encoder,
            'main_latent_compressor': main_latent_compressor,
            'signature_latent_compressors': signature_latent_compressors,
            'signature_regressors': signature_regressors,
            'pheno_latent_compressors': pheno_latent_compressors,
            'pheno_models': pheno_models,
            'decoder': decoder
        })

        if self.verbose:
            print("Model built:")
            print(self.model)

    def forward(self, batch,
                forward_signature=True, selected_signature=None,
                forward_pheno=True, selected_pheno=None,
                forward_main_latent=True, forward_reconstruction=True):
        """
        Forward extractor framework.

        :param batch: (torch.Tensor) gene expression tensors, shape should be (N,M), where N is number of cell, M is number of gene
        :param forward_signature: (bool) should signature supervision part be forwarded
        :param selected_signature: (list or None) list of selected signatures to be forwarded, None to forward all signatures
        :param forward_pheno: (bool) should phenotype supervision part be forwarded
        :param selected_pheno: (list or None) list of selected phenotypes to be forwarded, None to forward all phenotypes
        :param forward_main_latent: (bool) should main latent part be forwarded
        :param forward_reconstruction: (bool) should decoder be forwarded (decoder could be forwarded only when all latent dimensions are forwarded)
        :return: a dictionary containing results of forwarding
        """
        # Forward Pre Encoder
        x = self.model['pre_encoder'](batch)

        # Forward Main Latent
        lat_main = torch.Tensor()
        lat_all = torch.Tensor()
        if forward_main_latent:
            lat_main = self.model['main_latent_compressor'](x)
            lat_all = lat_main

        # Forward signature supervision
        lat_signature = dict()
        signature_out = dict()
        attached_signatures = list()
        if forward_reconstruction or forward_signature:
            if selected_signature is None:
                # Select all signature
                selected_signature = self.signature_config.keys()
            for cur_signature in selected_signature:
                model_details = self.signature_config[cur_signature].get('model')
                if model_details is not None:
                    if model_details.get('attach') == 'True':
                        attached_signatures.append(cur_signature)
                        continue
                lat_signature[cur_signature] = self.model['signature_latent_compressors'][cur_signature](x)
                lat_all = torch.cat((lat_all, lat_signature[cur_signature]), 1)
                signature_out[cur_signature] = self.model['signature_regressors'][cur_signature](
                    lat_signature[cur_signature])

        # Forward phenotype supervision
        lat_pheno = dict()
        pheno_out = dict()
        attached_phenos = list()
        if forward_reconstruction or forward_pheno:
            if selected_pheno is None:
                selected_pheno = self.pheno_config.keys()
            for cur_pheno in selected_pheno:
                model_details = self.pheno_config[cur_pheno].get('model')
                if model_details is not None:
                    if model_details.get('attach') == 'True':
                        attached_phenos.append(cur_pheno)
                        continue
                lat_pheno[cur_pheno] = self.model['pheno_latent_compressors'][cur_pheno](x)
                lat_all = torch.cat((lat_all, lat_pheno[cur_pheno]), 1)
                pheno_out[cur_pheno] = self.model['pheno_models'][cur_pheno](lat_pheno[cur_pheno])

        # Reconstruct input gene expression profiles
        re_x = None
        if forward_reconstruction:
            re_x = self.model['decoder'](lat_all)

        # Process attached signature/pheno tasks
        def handle_attach(model_details):
            if model_details['attach_to'] == 'main_lat':
                return lat_main
            elif model_details['attach_to'] == 'signature_lat':
                return lat_signature[model_details['attach_key']]
            elif model_details['attach_to'] == 'pheno_lat':
                return lat_pheno[model_details['attach_key']]
            elif model_details['attach_to'] == 'all_lat':
                return lat_all
            elif model_details['attach_to'] == 'multiple':
                lat_cur = torch.Tensor()
                for cur_attach in model_details['attach_key']:
                    if cur_attach['type'] == 'pheno':
                        lat_cur = torch.cat((lat_cur, lat_pheno[cur_attach['key']]), 1)
                    elif cur_attach['type'] == 'signature':
                        lat_cur = torch.cat((lat_cur, lat_signature[cur_attach['key']]), 1)
                    elif cur_attach['type'] == 'main':
                        lat_cur = torch.cat((lat_cur, lat_main), 1)
                return lat_cur

        # Attached signatures
        if forward_signature:
            for cur_signature in attached_signatures:
                model_details = self.signature_config[cur_signature].get('model')
                lat_signature[cur_signature] = handle_attach(model_details)
                signature_out[cur_signature] = self.model['signature_regressors'][cur_signature](lat_signature[cur_signature])

        # Attached phenos
        if forward_pheno:
            for cur_pheno in attached_phenos:
                model_details = self.pheno_config[cur_pheno].get('model')
                lat_pheno[cur_pheno] = handle_attach(model_details)
                pheno_out[cur_pheno] = self.model['pheno_models'][cur_pheno](lat_pheno[cur_pheno])

        return {
            'x': batch,
            'lat_main': lat_main,
            'lat_signature': lat_signature,
            'signature_out': signature_out,
            'lat_pheno': lat_pheno,
            'pheno_out': pheno_out,
            're_x': re_x,
            'lat_all': lat_all
        }




