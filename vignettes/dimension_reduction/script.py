import gpflow
import numpy as np
import pandas as pd

import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_probability as tfp

import gpflow
from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter
from typing import Optional

from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float
from gpflow.utilities.ops import pca_reduce
from gpflow.models.model import BayesianModel
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper



#import cop-e-cat dfs

Y = data_df[['ALBUMIN', 'ANION GAP',
       'BASE EXCESS', 'BICARBONATE', 'BILIRUBIN', 'CALCIUM',
       'CARBOXYHEMOGLOBIN', 'CHLORIDE', 'CREATININE', 'HEMATOCRIT',
       'HEMOGLOBIN', 'INSPIRED OXYGEN', 'INTERNATIONAL NORMALIZED RATIO',
       'LACTATE', 'METHEMOGLOBIN', 'PARTIAL THROMBOPLASTIN TIME', 'PCO2', 'PH',
       'PLATELETS', 'PO2', 'POTASSIUM', 'SODIUM', 'UREA NITROGEN',
       'URINE OUTPUT', 'WHITE BLOOD CELLS', 'FIO2', 'PEEP', 'OXYGEN (L)',
       'Respiratory Aids', 'Nonsteroidal Anti-inflammatory Agents (NSAIDs)',
       'Corticosteroids - Topical', 'Mineralocorticoids',
       'Glucocorticosteroids', 'Influenza Agents', 'Antiretrovirals',]]

# Y = Y[:300]
for col in Y.columns:
    Y[(Y < 0) | (Y > 1000)][col] = Y[(Y > 0) & (Y < 1000)][col].mean()
    
std = Y.std()
zero_std, = np.where(std == 0)
std[zero_std] = 1
Y = (Y - Y.mean()) / std

class tGPLVM(BayesianModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        mean_function=None,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        X_prior_mean=None,
        X_prior_var=None,
        degrees_of_freedom=3,
        ):
        """
        Initialise tGPLVM object. This method only works with a Gaussian likelihood.
        :param data: data matrix, size N (number of points) x D (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        super().__init__()
        
        num_data, self.num_latent_gps = X_data_mean.shape
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihoods.Gaussian()
        
        self.data = data_input_to_tensor(data)
        assert X_data_var.ndim == 2

        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.num_data = num_data
        self.output_dim = self.data.shape[-1]
        
        self.dof = degrees_of_freedom

        assert np.all(X_data_mean.shape == X_data_var.shape)
        assert X_data_mean.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert X_data_var.shape[0] == self.data.shape[0], "X var and Y must be same size."

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            Z = X_data_mean[xu_init]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent_gps

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data = self.data

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
                
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        
        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_data_var = (
            self.X_data_var
            if self.X_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.X_data_var)
        )
        NQ = to_default_float(tf.size(self.X_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        
        return bound
    
    def calc_num_latent_gps_from_data(self, data, kernel, likelihood):
        """
        Calculates the number of latent GPs required based on the data as well
        as the type of kernel and likelihood.
        """
        _, Y = data
        output_dim = Y.shape[-1]
        
        if isinstance(kernel, MultioutputKernel):
            # MultioutputKernels already have num_latent_gps attributes
            num_latent_gps = kernel.num_latent_gps
        elif isinstance(likelihood, SwitchedLikelihood):
            # the SwitchedLikelihood partitions/stitches based on the last
            # column in Y, but we should not add a separate latent GP for this!
            # hence decrement by 1
            num_latent_gps = output_dim - 1
            assert num_latent_gps > 0
        else:
            num_latent_gps = output_dim
        return num_latent_gps
    

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        """
        :param Xnew: points at which to predict
        """
        if full_output_cov:
            raise NotImplementedError

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))
        
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), axis=0)
                - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
    
    
    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """

        mean, cov = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        y = tfp.distributions.StudentT(self.df, mean, cov, name='StudentT')
        return y.sample(sample_shape=(num_samples))
    
    def predict_y(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Compute the mean and variance of the held-out data at the input points.
        """

        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)
    

    def predict_log_density(self, data, full_cov=False, full_output_cov=False):
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)


from sklearn.decomposition import PCA


latent_dim = 3  # number of latent dimensions
num_inducing = 20  # number of inducing pts
num_data = Y.shape[0]  # number of data points


pca = PCA(n_components = latent_dim)
X_mean_init = tf.convert_to_tensor(pca.fit_transform(Y))
X_var_init = tf.ones((num_data, latent_dim), dtype=default_float())


np.random.seed(1)  # for reproducibility

batchsize = 30

opt = gpflow.optimizers.Scipy()
maxiter = ci_niter(100)
params = None

for i in range(0, Y.shape[0] - batchsize, batchsize):
    print(i)
    inducing_variable = tf.convert_to_tensor(
        np.random.permutation(X_mean_init[i:i+batchsize].numpy())[:num_inducing], dtype=default_float()
    )
    
    lengthscales = tf.convert_to_tensor([1.0] * latent_dim, dtype=default_float())
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)

    
    gplvm = tGPLVM(Y[i:i+batchsize],
                    X_data_mean=X_mean_init[i:i+batchsize],
                    X_data_var=X_var_init[i:i+batchsize],
                    kernel=kernel,
                    inducing_variable=inducing_variable)
    if params is not None:
        gpflow.utilities.multiple_assign(gplvm, params)
    else:
        gplvm.likelihood.variance.assign(0.01)

    opt.minimize(gplvm.training_loss,
                variables=gplvm.trainable_variables,
                options=dict(maxiter=maxiter))
    params = gpflow.utilities.parameter_dict(gplvm)

# visualization 
gplvm_X_mean = gplvm.X_data_mean.numpy()

f, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].scatter(X_mean_init[i:i+batchsize, 0], X_mean_init[i:i+batchsize, 1])
ax[1].scatter(gplvm_X_mean[:, 0], gplvm_X_mean[:, 1])
ax[0].set_title("PCA")
ax[1].set_title("Bayesian GPLVM")

#save to output directory 
