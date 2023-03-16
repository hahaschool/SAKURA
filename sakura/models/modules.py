import torch.nn as nn


def modulebuilder(cfg):
    ret = list() #nn.ModuleList()
    cur_dim=cfg['in_dim']
    for cur_module in cfg:
        if cur_module['type'] == 'Linear':
            ret.append(nn.Linear(in_features=cur_module['in_dim'], out_features=cur_module['out_dim']))
            cur_dim = cur_module['out_dim']
        elif cur_module['type'] == 'Dropout':
            ret.append(nn.Dropout(p=cur_module.get('p')))
        elif cur_module['type'] == 'ReLU':
            ret.append(nn.ReLU())
        elif cur_module['type'] == 'CELU':
            ret.append(nn.CELU())
        elif cur_module['type'] == 'Softmax':
            ret.append(nn.Softmax())
        elif cur_module['type'] == 'LogSoftmax':
            ret.append(nn.LogSoftmax)

    return ret



# class GELU(nn.Module):


class FCDecoder(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_neurons=50, hidden_layers=3,
                 output_activation_function='identity',
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5,
                 config=None):
        super(FCDecoder, self).__init__()
        self.model_list = nn.ModuleList()
        self.config = config
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.config is None:
            # Default 3 hidden layer structure
            # Input --> Linear --> CELU --> Linear --> CELU --> Linear --> Output
            if dropout and dropout_input:
                self.model_list.append(nn.Dropout(p=dropout_input_p))

            if self.hidden_layers == 1:
                # Default is 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
            elif self.hidden_layers > 1:
                if type(hidden_neurons) is int:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                        self.model_list.append(nn.CELU())
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))

                elif type(hidden_neurons) is list:
                    cur_hidden_i = 0
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons[cur_hidden_i]))
                    self.model_list.append(nn.CELU())
                    cur_hidden_i += 1
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.hidden_neurons[cur_hidden_i]))
                        self.model_list.append(nn.CELU())
                        cur_hidden_i += 1
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.output_dim))
                else:
                    raise ValueError("Either specify an integer or a list of integers to hidden_neurons.")

            if output_activation_function == 'relu':
                self.model_list.append(nn.ReLU())
            elif output_activation_function == 'softmax':
                self.model_list.append(nn.Softmax())
            elif output_activation_function != 'identity':
                raise NotImplementedError('Unsupported activation function')

        else:
            self.model_list = modulebuilder(config)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x

class FCPreEncoder(nn.Module):
    # Input --> Linear --> CELU --> Linear --> CELU --> Output
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_neurons=None, hidden_layers=2,
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5,
                 config=None):
        super(FCPreEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers

        if config is None:
            # Default 2 hidden layer structure
            # Input --> Linear --> CELU --> Linear --> CELU --> Output (Low-dim compressor expected)
            # Dropout input data
            if dropout and dropout_input:
                self.model_list.append(nn.Dropout(p=dropout_input_p))
            if self.hidden_layers == 1:
                # 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
                self.model_list.append(nn.CELU())
            elif self.hidden_layers > 1:
                if type(hidden_neurons) is int:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                        self.model_list.append(nn.CELU())
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))


                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
                    self.model_list.append(nn.CELU())
                elif type(hidden_neurons) is list:
                    cur_hidden_i = 0
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons[cur_hidden_i]))
                    self.model_list.append(nn.CELU())
                    cur_hidden_i += 1
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.hidden_neurons[cur_hidden_i]))
                        self.model_list.append(nn.CELU())
                        cur_hidden_i += 1
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))


                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.output_dim))
                    self.model_list.append(nn.CELU())
                else:
                    raise ValueError("[FCPreEncoder] Either specify an integer or a list of integers to hidden_neurons.")

            else:
                raise ValueError("[FCPreEncoder] The number of hidden layer of FCCompressor should be 1, or larger than 1")
        else:
            self.model_list = modulebuilder(self.config)
    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x


class FCCompressor(nn.Module):
    """
    Simply used to compress outputs from pre-encoder to a lower dimension
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_neurons: int = 50, hidden_layers: int = 1,
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5,
                 config=None):
        super(FCCompressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        if self.config is None:
            if self.hidden_layers == 1:
                # Default is 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
                self.model_list.append(nn.CELU())
            elif self.hidden_layers > 1:
                if type(hidden_neurons) is int:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                        self.model_list.append(nn.CELU())

                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
                    self.model_list.append(nn.CELU())
                elif type(hidden_neurons) is list:
                    cur_hidden_i = 0
                    self.model_list.append(
                        nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons[cur_hidden_i]))
                    self.model_list.append(nn.CELU())
                    cur_hidden_i += 1

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.hidden_neurons[cur_hidden_i]))
                        self.model_list.append(nn.CELU())
                        cur_hidden_i += 1

                    self.model_list.append(
                        nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.output_dim))
                    self.model_list.append(nn.CELU())
                else:
                    raise ValueError("Either specify an integer or a list of integers to hidden_neurons.")

            else:
                raise ValueError("The number of hidden layer of FCCompressor should be 1, or larger than 1")
        else:
            self.model_list=modulebuilder(self.config)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x


class FCClassifier(nn.Module):
    """
    Module used for supervising cell labels
    Use entire latent space as input, or designated dimension(s)
    Training goal is to predict cell labels (e.g. cell type, group)
    """

    def __init__(self, input_dim, output_dim,
                 hidden_neurons=5, hidden_layers=None,
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5,
                 config=None):
        super(FCClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        self.model_list = nn.ModuleList()
        if self.config is None:
            if hidden_layers is None:
                # Legacy mode:
                # Input --> Linear --> CELU --> Linear --> CELU --> Linear --> LogSoftmax --> Output
                # Dropout input data
                if dropout and dropout_input:
                    self.model_list.append(nn.Dropout(p=dropout_input_p))
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())

                # Dropout hidden layer activations
                if dropout and dropout_hidden:
                    self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())

                # Dropout hidden layer activations
                if dropout and dropout_hidden:
                    self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
            elif type(hidden_layers) is int:
                # Dropout input data
                if dropout and dropout_input:
                    self.model_list.append(nn.Dropout(p=dropout_input_p))

                if hidden_layers == 1:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
                else:
                    if type(self.hidden_neurons) is int:
                        # Backcompat, fixed number of hidden neurons
                        self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                        self.model_list.append(nn.CELU())
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                        # If more than 2 layers requested
                        for i in range(self.hidden_layers - 2):
                            self.model_list.append(
                                nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                            self.model_list.append(nn.CELU())
                            if dropout and dropout_hidden:
                                self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                        self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
                    elif type(self.hidden_neurons) is list:
                        cur_hidden_i = 0
                        self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons[cur_hidden_i]))
                        self.model_list.append(nn.CELU())
                        cur_hidden_i += 1
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                        # If more than 2 layers requested
                        for i in range(self.hidden_layers - 2):
                            self.model_list.append(
                                nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.hidden_neurons[cur_hidden_i]))
                            self.model_list.append(nn.CELU())
                            cur_hidden_i += 1
                            if dropout and dropout_hidden:
                                self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.output_dim))
                    else:
                        raise ValueError(
                            '[FCClassifier] hidden_neurons should be int for legacy, fixed hidden mode, or list of int for flex mode')
            else:
                raise ValueError('[FCClassifier] hidden_layers should be None for legacy mode, or int for flex mode')

            # For classifier, the activation function of output layer is always Logsoftmax
            self.model_list.append(nn.LogSoftmax(dim=1))
        else:
            self.model_list = modulebuilder(self.config)
        self.model = nn.Sequential(self.model_list)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x

class GeneralFCLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_layers=None, hidden_neurons=None,
                 hidden_activation_function='CELU',
                 output_activation_function='identity',
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5):

        super(GeneralFCLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p

        # TODO: impl a generalized FC layer and replace current modules with the general version

        raise NotImplementedError


class FCRegressor(nn.Module):
    """
    Model used for supervising expression levels for selected genes
    Use entire latent space as input, or designated dimension(s)
    """

    def __init__(self, input_dim, output_dim, config=None,
                 hidden_neurons=5, hidden_layers=None,
                 output_activation_function='identity',
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5):
        super(FCRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        if self.config is None:
            if hidden_layers is None:
                # Legacy model
                # Input --> Linear --> CELU --> Linear --> CELU --> Linear --> ReLU --> Output
                if dropout and dropout_input:
                    self.model_list.append(nn.Dropout(p=dropout_input_p))

                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())
                if dropout and dropout_hidden:
                    self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())
                if dropout and dropout_hidden:
                    self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
            elif type(hidden_layers) is int:

                # Dropout input data
                if dropout and dropout_input:
                    self.model_list.append(nn.Dropout(p=dropout_input_p))

                if hidden_layers == 1:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))

                elif type(hidden_neurons) is int:
                    self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                        self.model_list.append(nn.CELU())
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
                elif type(hidden_neurons) is list:
                    cur_hidden_i = 0
                    self.model_list.append(
                        nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons[cur_hidden_i]))
                    self.model_list.append(nn.CELU())
                    cur_hidden_i += 1
                    if dropout and dropout_hidden:
                        self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    # If more than 2 layers requested
                    for i in range(self.hidden_layers - 2):
                        self.model_list.append(
                            nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.hidden_neurons[cur_hidden_i]))
                        self.model_list.append(nn.CELU())
                        cur_hidden_i += 1
                        if dropout and dropout_hidden:
                            self.model_list.append(nn.Dropout(p=dropout_hidden_p))

                    self.model_list.append(
                        nn.Linear(in_features=self.hidden_neurons[cur_hidden_i-1], out_features=self.output_dim))
            else:
                raise ValueError('[FCRegressor] hidden_layers should be None for legacy mode, or int for flex mode')
            if output_activation_function == 'relu':
                self.model_list.append(nn.ReLU())
            elif output_activation_function == 'softmax':
                self.model_list.append(nn.Softmax())
            elif output_activation_function == 'sigmoid':
                self.model_list.append(nn.Sigmoid())
            elif output_activation_function != 'identity':
                raise NotImplementedError('Unsupported activation function')
        else:
            self.model_list = modulebuilder(self.config)
        self.model = nn.Sequential(self.model_list)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x

class LinClassifier(nn.Module):
    """
    Use single linear layer and softmax activation function to do classification
    Useful when simple and linear structure is expected from certain laten dimension
    Input --> Linear --> LogSoftmax --> Output
    """
    def __init__(self, input_dim, output_dim, config=None):
        super(LinClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        if self.config is None:
            # Input --> Linear --> Softmax --> Output
            self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
            self.model_list.append(nn.LogSoftmax(dim=1))
        else:
            self.model_list = modulebuilder(self.config)
        self.model = nn.Sequential(self.model_list)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x

class LinRegressor(nn.Module):
    """
    Use simple linear regressor to predict selected expression levels
    Input is entire latent space, or designated deiension(s)
    Expected to make latent space aligned along linear structure
    """

    def __init__(self, input_dim, output_dim, config=None, output_activation_function='identity'):
        super(LinRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        if self.config is None:
            # Input --> Linear --> Output
            self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
            if output_activation_function == 'relu':
                self.model_list.append(nn.ReLU())
            elif output_activation_function == 'softmax':
                self.model_list.append(nn.Softmax())
            elif output_activation_function != 'identity':
                raise NotImplementedError('Unsupported activation function')
        else:
            self.model_list = modulebuilder(self.config)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x
