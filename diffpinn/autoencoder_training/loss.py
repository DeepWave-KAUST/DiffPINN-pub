"""
Description:
This module defines the loss function for a Wasserstein Autoencoder with Maximum Mean Discrepancy (WAE-MMD) regularization.
It computes a combined loss that includes a reconstruction term (mean squared error) and an MMD term between the latent distribution and a prior.
The MMD term can be computed using either an RBF or an Inverse Multiquadric (IMQ) kernel.
"""

import torch
from torch import nn
import torch.nn.functional as F

class WAE_MMD_LOSS(nn.Module):
    def __init__(self, latent_dim, reg_weight, reconstruction_loss_scale, kernel_choice, 
                 kernel_bandwidth, scales=[0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]):
        """
        Initializes the WAE_MMD_LOSS module.

        Arguments:
        - latent_dim: Dimensionality of the latent space.
        - reg_weight: Regularization weight for the MMD loss term.
        - reconstruction_loss_scale: Scaling factor for the reconstruction loss.
        - kernel_choice: Choice of kernel to compute MMD ('rbf' for Gaussian RBF or other for IMQ).
        - kernel_bandwidth: Bandwidth parameter for the kernel.
        - scales: List of scales to use in the IMQ kernel. Defaults to a preset list if not provided.
        """
        super(WAE_MMD_LOSS, self).__init__()
        self.reg_weight = reg_weight
        self.latent_dim = latent_dim
        self.reconstruction_loss_scale = reconstruction_loss_scale
        self.kernel_choice = kernel_choice
        self.kernel_bandwidth = kernel_bandwidth
        self.scales = scales if scales is not None else [1.0]

    def loss_function(self, recon_x, x, z, z_prior):
        """
        Computes the combined loss consisting of reconstruction loss and MMD loss.

        Arguments:
        - recon_x: Reconstructed output from the autoencoder.
        - x: Original input.
        - z: Latent representation produced by the encoder.
        - z_prior: Sampled latent codes from the prior distribution.

        Returns:
        - A tuple containing:
            1. Total loss: reconstruction loss (mean MSE) + reg_weight * MMD loss.
            2. Mean reconstruction loss.
            3. MMD loss.
        """
        N = z.shape[0]  # Batch size

        # Compute reconstruction loss using mean squared error (MSE)
        # Reshape tensors to 2D (batch_size x -1) so that the loss is computed per sample
        recon_loss = self.reconstruction_loss_scale * F.mse_loss(
            recon_x.reshape(x.shape[0], -1), 
            x.reshape(x.shape[0], -1), 
            reduction="none"
        ).sum(dim=-1)

        # Compute pairwise kernel matrices based on the chosen kernel type
        if self.kernel_choice == "rbf":
            # Using the RBF kernel
            k_z = self.rbf_kernel(z.reshape(x.shape[0], -1), z.reshape(x.shape[0], -1))
            k_z_prior = self.rbf_kernel(z_prior.reshape(x.shape[0], -1), z_prior.reshape(x.shape[0], -1))
            k_cross = self.rbf_kernel(z.reshape(x.shape[0], -1), z_prior.reshape(x.shape[0], -1))
        else:
            # Using the IMQ kernel by default
            k_z = self.imq_kernel(z.reshape(x.shape[0], -1), z.reshape(x.shape[0], -1))
            k_z_prior = self.imq_kernel(z_prior.reshape(x.shape[0], -1), z_prior.reshape(x.shape[0], -1))
            k_cross = self.imq_kernel(z.reshape(x.shape[0], -1), z_prior.reshape(x.shape[0], -1))

        # Compute MMD loss components:
        # Remove the diagonal elements (self-similarities) for k_z and k_z_prior.
        mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
        # Average cross-kernel values between z and z_prior
        mmd_cross = k_cross.sum() / (N**2)

        # Combine the MMD terms to form the final MMD loss
        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

        # Return the total loss, along with individual components for logging
        return (
            recon_loss.mean(dim=0) + self.reg_weight * mmd_loss,
            recon_loss.mean(dim=0),
            mmd_loss,
        )

    def imq_kernel(self, z1, z2):
        """
        Computes the Inverse Multiquadric (IMQ) kernel matrix between two sets of vectors.

        Arguments:
        - z1: Tensor of shape [batch_size, latent_dim].
        - z2: Tensor of shape [batch_size, latent_dim].

        Returns:
        - A kernel matrix of shape [batch_size x batch_size] with pairwise IMQ kernel values.
        """
        # Base constant for the kernel computed from latent dimension and bandwidth
        Cbase = 2.0 * self.latent_dim * self.kernel_bandwidth**2

        k = 0
        # Sum the contributions of kernels at multiple scales
        for scale in self.scales:
            C = scale * Cbase
            # Compute pairwise distances between elements in z1 and z2 and accumulate the kernel values
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)
        return k

    def rbf_kernel(self, z1, z2):
        """
        Computes the Radial Basis Function (RBF) kernel matrix between two sets of vectors.

        Arguments:
        - z1: Tensor of shape [batch_size, latent_dim].
        - z2: Tensor of shape [batch_size, latent_dim].

        Returns:
        - A kernel matrix of shape [batch_size x batch_size] with pairwise RBF kernel values.
        """
        # Constant term for the RBF kernel based on latent dimension and bandwidth
        C = 2.0 * self.latent_dim * self.kernel_bandwidth**2
        # Compute the pairwise RBF kernel values using the exponential of the negative squared distance
        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)
        return k

    def forward(self, x, recon_x, latent):
        """
        Forward method to compute the loss given the original input, reconstructed output, and latent representation.

        Arguments:
        - x: Original input tensor.
        - recon_x: Reconstructed tensor from the autoencoder.
        - latent: Latent representation output by the encoder.

        Returns:
        - A tuple containing the total loss, reconstruction loss, and MMD loss.
        """
        # Sample a prior latent vector from a standard normal distribution with the same shape as latent
        latent_prior = torch.randn_like(latent, device=x.device)
        # Compute the losses using the loss_function helper method
        loss, recon_loss, mmd_loss = self.loss_function(recon_x, x, latent, latent_prior)
        return loss, recon_loss, mmd_loss
