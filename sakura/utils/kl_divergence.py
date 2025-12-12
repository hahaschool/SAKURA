"""
KL divergence for loss computation
"""

import torch

class KLDivergence(object):
    """
    Computes KL Divergence between encoded samples and target distribution
    Assumes both distributions are multivariate Gaussian with diagonal covariance
    """

    def __init__(self, eps=1e-8):
        """
        Initialize with a small epsilon for numerical stability

        :param eps: Small value to prevent numerical issues
        :type eps: float
        """
        self.eps = eps

    def _kl_divergence(self, encoded_samples, distribution_samples, target):
        """
        Compute KL divergence between two Gaussian distributions

        :param encoded_samples: Samples from encoded distribution
        :type encoded_samples: torch.Tensor
        :param distribution_samples: Samples from target distribution
        :type distribution_samples: torch.Tensor

        :return: KL divergence between the two distributions
        :rtype: torch.Tensor
        """
        # Compute mean and variance for both distributions
        mu_p = torch.mean(encoded_samples, dim=0)
        var_p = torch.var(encoded_samples, dim=0, unbiased=False) + self.eps

        mu_q = torch.mean(distribution_samples, dim=0)
        var_q = torch.var(distribution_samples, dim=0, unbiased=False) + self.eps

        # KL divergence formula for two univariate Gaussians
        if target == 'gaussian':
            kl = 0.5 * (torch.log(var_q) - torch.log(var_p) +
                        (var_p + (mu_p - mu_q) ** 2) / var_q - 1)
        elif target == 'gaussian_mixture':
            raise NotImplementedError
        else:
            raise ValueError

        # Sum across all latent dimensions
        return torch.sum(kl)

    def kl_divergence(self, encoded_samples,
                      distribution_fn,
                      target,
                      device='cpu'):
        """
        Compute KL divergence between encoded samples and distribution function samples

        :param encoded_samples: Samples from encoded distribution
        :type encoded_samples: torch.Tensor
        :param distribution_fn: Function that generates drawn distribution samples (args: batch_size, n_dim)
        :type distribution_fn: Callable
        :param device: torch computation device, defaults to 'cpu'
        :type device: Literal['cpu', 'cuda'], optional

        :return: KL divergence between the distributions
        :rtype: torch.Tensor
        """
        # Derive batch size and latent dimension from encoded samples
        batch_size = encoded_samples.size(0)
        latent_dim = encoded_samples.size(1)

        # Draw samples from target distribution
        z = distribution_fn(batch_size, n_dim=latent_dim).to(device)

        # Compute KL divergence
        kl = self._kl_divergence(encoded_samples, z, target)
        return kl

    def __call__(self, encoded_samples,
                 distribution_fn,
                 device='cpu'):
        """
        Callable interface for KL divergence computation

        :param encoded_samples: Samples from encoded distribution
        :type encoded_samples: torch.Tensor
        :param distribution_fn: Function that generates drawn distribution samples (args: batch_size, n_dim)
        :type distribution_fn: Callable
        :param device: torch computation device, defaults to 'cpu'
        :type device: Literal['cpu', 'cuda'], optional

        :return: KL divergence between the distributions
        :rtype: torch.Tensor
        """
        return self.kl_divergence(encoded_samples,
                                  distribution_fn,
                                  target='uniform',
                                  device=device)
