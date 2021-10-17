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
    def __init__(self, input_dim, output_dim, hidden_neurons=50, hidden_layers=3, output_activation_function='identity', config=None):
        super(FCDecoder, self).__init__()
        self.model_list = nn.ModuleList()
        self.config = config
        if self.config is None:
            # Default 3 hidden layer structure
            # Input --> Linear --> CELU --> Linear --> CELU --> Linear --> Output
            # The difference btw FCPreEncoder/FCCompressor is default layers and no CELU activations
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_neurons = hidden_neurons
            self.hidden_layers = hidden_layers
            if self.hidden_layers == 1:
                # Default is 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
            elif self.hidden_layers > 1:
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())

                # If more than 2 layers requested
                for i in range(self.hidden_layers - 2):
                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))

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
    def __init__(self, input_dim: int, output_dim: int, hidden_neurons: int = 50, hidden_layers: int = 2, config=None):
        super(FCPreEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        if config is None:
            # Default 2 hidden layer structure
            # Input --> Linear --> CELU --> Linear --> CELU --> Output (Low-dim compressor expected)
            self.hidden_neurons = hidden_neurons
            self.hidden_layers = hidden_layers

            if self.hidden_layers == 1:
                # Default is 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
                self.model_list.append(nn.CELU())
            elif self.hidden_layers > 1:
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())

                # If more than 2 layers requested
                for i in range(self.hidden_layers - 2):
                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
                self.model_list.append(nn.CELU())
            else:
                raise ValueError("The number of hidden layer of FCCompressor should be 1, or larger than 1")
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

    def __init__(self, input_dim: int, output_dim: int, hidden_neurons: int = 50, hidden_layers: int = 1, config=None):
        super(FCCompressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        if self.config is None:
            self.hidden_neurons = hidden_neurons
            self.hidden_layers = hidden_layers

            if self.hidden_layers == 1:
                # Default is 1 layer structure
                # Input --> Output (latent transformation)
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.output_dim))
            elif self.hidden_layers > 1:
                self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
                self.model_list.append(nn.CELU())

                # If more than 2 layers requested
                for i in range(self.hidden_layers - 2):
                    self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
                    self.model_list.append(nn.CELU())

                self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
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
                 hidden_neurons=5,
                 dropout=False,
                 dropout_input=False, dropout_input_p=0.5,
                 dropout_hidden=False, dropout_hidden_p=0.5,
                 config=None):
        super(FCClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.hidden_neurons = hidden_neurons
        self.dropout_input = dropout_input
        self.dropout_input_p = dropout_input_p
        self.dropout_hidden = dropout_hidden
        self.dropout_hidden_p = dropout_hidden_p
        self.model_list = nn.ModuleList()
        if self.config is None:
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
            self.model_list.append(nn.LogSoftmax(dim=1))
        else:
            self.model_list = modulebuilder(self.config)
        self.model = nn.Sequential(self.model_list)

    def forward(self, x):
        for cur_model in self.model_list:
            x = cur_model(x)
        return x

class FCRegressor(nn.Module):
    """
    Model used for supervising expression levels for selected genes
    Use entire latent space as input, or designated dimension(s)
    """

    def __init__(self, input_dim, output_dim, config=None, hidden_neurons=5, output_activation_function='identity'):
        super(FCRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.model_list = nn.ModuleList()
        if self.config is None:
            # Input --> Linear --> CELU --> Linear --> CELU --> Linear --> ReLU --> Output
            self.hidden_neurons = hidden_neurons
            self.model_list.append(nn.Linear(in_features=self.input_dim, out_features=self.hidden_neurons))
            self.model_list.append(nn.CELU())
            self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.hidden_neurons))
            self.model_list.append(nn.CELU())
            self.model_list.append(nn.Linear(in_features=self.hidden_neurons, out_features=self.output_dim))
            if output_activation_function == 'relu':
                self.model_list.append(nn.ReLU())
            elif output_activation_function == 'softmax':
                self.model_list.append(nn.Softmax())
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
