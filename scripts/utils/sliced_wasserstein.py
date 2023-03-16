import numpy as np
import torch

class SlicedWasserstein(object):
    def rand_projections(self, embedding_dim, num_samples=50):
        """This function generates `num_samples` random samples from the latent space's unit sphere.

            Args:
                embedding_dim (int): embedding dimensionality
                num_samples (int): number of random projection samples

            Return:
                torch.Tensor: tensor of size (num_samples, embedding_dim)
        """
        projections = [w / np.sqrt((w ** 2).sum())  # L2 normalization
                       for w in np.random.normal(size=(num_samples, embedding_dim))]
        projections = np.asarray(projections)
        return torch.from_numpy(projections).float()

    def _sliced_wasserstein_distance(self, encoded_samples,
                                     distribution_samples,
                                     num_projections=50,
                                     p=2,
                                     device='cpu'):
        """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

            Args:
                encoded_samples (toch.Tensor): tensor of encoded training samples
                distribution_samples (torch.Tensor): tensor of drawn distribution training samples
                num_projections (int): number of projections to approximate sliced wasserstein distance
                p (int): power of distance metric
                device (torch.device): torch device (default 'cpu')

            Return:
                torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
        """
        # derive latent space dimension size from random samples drawn from latent prior distribution
        embedding_dim = distribution_samples.size(1)
        # generate random projections in latent space
        projections = self.rand_projections(embedding_dim, num_projections).to(device)
        # calculate projections through the encoded samples
        encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
        # calculate projections through the prior distribution random samples
        distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
        # calculate the sliced wasserstein distance by
        # sorting the samples per random projection and
        # calculating the difference between the
        # encoded samples and drawn random samples
        # per random projection
        wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
        # distance between latent space prior and encoded distributions
        # power of 2 by default for Wasserstein-2
        wasserstein_distance = torch.pow(wasserstein_distance, p)
        # approximate mean wasserstein_distance for each projection
        return wasserstein_distance.mean()

    def sliced_wasserstein_distance(self, encoded_samples,
                                    distribution_fn,
                                    num_projections=50,
                                    p=2,
                                    device='cpu'):
        """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

            Args:
                encoded_samples (tocrh.Tensor): tensor of encoded training samples
                distribution_fn (func): a function to generate distributions
                num_projections (int): number of projections to approximate sliced wasserstein distance
                p (int): power of distance metric
                device (torch.device): torch device (default 'cpu')

            Return:
                torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
        """
        # derive batch size from encoded samples
        batch_size = encoded_samples.size(0)
        latent_dim = encoded_samples.size(1)
        # draw random samples from latent space prior distribution
        z = distribution_fn(batch_size, n_dim=latent_dim).to(device)
        # approximate mean wasserstein_distance between encoded and prior distributions
        # for each random projection
        swd = self._sliced_wasserstein_distance(encoded_samples, z,
                                                num_projections, p, device)
        return swd

    def __call__(self, encoded_samples,
                 distribution_fn,
                 num_projections=50,
                 p=2,
                 device='cpu'):
        return self.sliced_wasserstein_distance(encoded_samples,
                                                distribution_fn,
                                                num_projections=50,
                                                p=2,
                                                device='cpu')
