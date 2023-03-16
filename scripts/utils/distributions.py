from math import sqrt, sin, cos

import numpy as np
import torch
from sklearn.datasets import make_circles


def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return torch.from_numpy(z).type(torch.FloatTensor)

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)
    return torch.from_numpy(z).type(torch.FloatTensor)

def rand_cirlce2d(batch_size):
    """ This function generates 2D samples from a filled-circle distribution in a 2-dimensional space.

        Args:
            batch_size (int): number of batch samples

        Return:
            torch.Tensor: tensor of size (batch_size, 2)
    """
    r = np.random.uniform(size=(batch_size))
    theta = 2 * np.pi * np.random.uniform(size=(batch_size))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.array([x, y]).T
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_ring2d(batch_size, n_dim=2):
    """ This function generates 2D samples from a hollowed-cirlce distribution in a 2-dimensional space.

        Args:
            batch_size (int): number of batch samples

        Return:
            torch.Tensor: tensor of size (batch_size, 2)
    """
    if n_dim != 2:
        raise NotImplementedError

    circles = make_circles(2 * batch_size, noise=.01)
    z = np.squeeze(circles[0][np.argwhere(circles[1] == 0), :])
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_uniform(batch_size, n_dim=2, low = -1., high = 1.,
                 n_labels=1, label_offsets=None, label_indices=None) -> torch.Tensor:
    """
    This function generates samples from a uniform distribution.
    :param batch_size (int): number of batch samples
    :param n_dim (int):  number of latent dimension
    :param low: lower bound of generated uniform distribution (for each dimension)
    :param high: upper bound of generated uniform distribution (for each dimension)
    :param n_labels: number of labels to consider in supervision, when n_labels=1, supervision is off
    :param label_offsets: offsets for different labels, for 1-d, could be a list of list (of floats, to build tensor), or list of floats, for more than 1-d, could be a list of list
    :return: torch.Tensor: tensor of size (batch_size, n_dim)
    """
    z = np.random.uniform(size=(batch_size, n_dim), low = low, high = high)

    if n_labels > 1:
        idx = np.array(label_indices, dtype=np.integer)
        offset = np.array([label_offsets[i] for i in idx]).reshape((batch_size, n_dim))
        z += offset


    return torch.from_numpy(z).type(torch.FloatTensor)


def rand(dim_size):
    def _rand(batch_size):
        return torch.rand((batch_size, dim_size))
    return _rand


def randn(dim_size):
    def _randn(batch_size):
        return torch.randn((batch_size, dim_size))
    return _randn
