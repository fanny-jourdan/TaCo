import numpy as np
from abc import ABC, abstractmethod
import scipy


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    @staticmethod
    def masks_dim(masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    @staticmethod
    def split_abc(outputs, nb_design, nb_dim):
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).
        nb_dim
          Number of dimensions to estimate.

        Returns
        -------
        a
          The results for the sample points in matrix A.
        b
          The results for the sample points in matrix B.
        c
          The results for the sample points in matrix C.
        """
        sampling_a = outputs[:nb_design]
        sampling_b = outputs[nb_design:nb_design*2]
        replication_c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return sampling_a, sampling_b, replication_c

    @staticmethod
    def post_process(stis, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        stis
          Total order Sobol' indices, one for each dimensions.
        masks
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis
          Total order Sobol' indices after post processing.
        """
        stis = np.array(stis, np.float32)
        return stis.reshape(masks.shape[1:])

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.

    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            np.sum((sampling_a - replication_c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)


class Sampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    def build_replicated_design(sampling_a, sampling_b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        sampling_a
          The masks values for the sampling matrix A.
        sampling_b
          The masks values for the sampling matrix B.

        Returns
        -------
        replication_c
          The new replicated design matrix C generated from A & B.
        """
        replication_c = np.array([sampling_a.copy() for _ in range(sampling_a.shape[-1])])
        for i in range(len(replication_c)):
            replication_c[i, :, i] = sampling_b[:, i]

        replication_c = replication_c.reshape((-1, sampling_a.shape[-1]))

        return replication_c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()


class ScipySampler(Sampler):
    """
    Base class based on Scipy qmc module for replicated design sampling.
    """

    def __init__(self):
        try:
            self.qmc = scipy.stats.qmc # pylint: disable=E1101
        except AttributeError as err:
            raise ModuleNotFoundError("Xplique need scipy>=1.7 to use this sampling.") from err


class ScipySobolSequence(ScipySampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Sobol(dimension*2, scramble=False)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)
