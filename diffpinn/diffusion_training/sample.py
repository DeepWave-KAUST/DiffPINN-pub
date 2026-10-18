"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description:
This script performs latent diffusion sampling using a pre-trained diffusion model, autoencoder, and PINN network.
It loads the trained diffusion model and autoencoder weights, and then samples latent representations for given test data.
The predicted outputs, intermediate predictions, and PDE losses (before and after diffusion) are saved in MATLAB (.mat) format.
"""

import argparse 
import os 
import numpy as np 
import torch as th  
import torch.nn.functional as F
import torch.distributed as dist 
from scipy.ndimage import gaussian_filter
from code.datasets import _list_image_files_recursively, normalizer_vel  # Custom dataset utilities
import scipy.io as sio  # For loading and saving MATLAB files
from scipy.ndimage import gaussian_filter  # Re-imported (can be removed if duplicate)
from code.train_util import parse_dataname_from_filename  # Utility for parsing data names from filenames
import matplotlib.pyplot as plt  # For plotting (if needed)
from code import logger  # Custom logging utility to record messages
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,  # Provides default parameters for model and diffusion
    create_model_and_diffusion,    # Function to create the diffusion model and process
    add_dict_to_argparser,         # Utility to add dictionary parameters to argparse parser
    args_to_dict,                  # Converts argparse arguments to a dictionary
)
from code.pinnmodel import PINN  # Import the PINN model defined in pinnmodel.py
from code.encoder import large  # Import the large autoencoder from encoder.py
import math  # For mathematical operations

def main():
    # Parse command-line arguments
    args = create_argparser().parse_args()

    # Define the device for computation (using GPU)
    device = th.device('cuda')
    
    # Configure logger for displaying information
    logger.configure()

    # Set up the output directory based on sampling method and PDE guidance settings
    if not args.use_ddim:
        if args.pde_guide:
            dir_output = f'./output/ddpm/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output/ddpm/usepde_{args.pde_guide}/'
    else:
        if args.pde_guide:
            dir_output = f'./output/{args.timestep_respacing}/usepde_{args.pde_guide}_scale{args.scale_factor}/'
        else:
            dir_output = f'./output/{args.timestep_respacing}/usepde_{args.pde_guide}/'
    # Create the output directory if it does not exist
    os.makedirs(dir_output, exist_ok=True)

    logger.log("creating model and diffusion...")
    # Create the diffusion model and process using default and command-line parameters
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # Load the pre-trained model state from the checkpoint file
    model.load_state_dict(
        th.load(f'{args.diff_path}', map_location="cpu")
    )
    # Move the model to the specified device (GPU) and set it to evaluation mode
    model.to(device)
    model.eval()

    # Choose the sampling function based on the use_ddim flag
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # ------------------ Autoencoder Initialization ------------------
    # Instantiate the autoencoder with specified noise factors and move it to the device
    AE = large(input_noise_factor=args.input_noise_factor, latent_noise_factor=args.latent_noise_factor).to(device)
    # Load the pre-trained autoencoder weights from the checkpoint file
    AE.load_state_dict(th.load(args.ae_path, map_location=device))

    # ------------------ PINN Initialization ------------------
    # Define network architecture parameters for PINN
    input_dim = 3
    output_dim = 2
    # Define the number of neurons in each layer (including input and output)
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]
    # Instantiate the PINN model and move it to the device
    pinn = PINN(neurons).to(device)
    # Store the shapes of the PINN parameters (for later use in functional forward pass)
    params_shapes = [p.size() for p in pinn.parameters()]

    # Calculate angular frequency omega based on the provided frequency argument
    omega = 2 * math.pi * args.freq

    logger.log("sampling...")

    data_list = ['CurveVelA', 'FlatFaultA', 'Layer']
    for fn in data_list:
        # Load individual test dataset MATLAB file
        dict = sio.loadmat(f'../../dataset/pinntest/{fn}.mat')
        # Extract and process velocity (v), coordinates, and wavefield data
        v = np.squeeze(np.sqrt(1/dict['m_train']))
        x_coords = np.squeeze(dict['x_train'])
        z_coords = np.squeeze(dict['z_train'])
        sx_coords = np.squeeze(dict['sx_train'])
        u0_real = dict['u0_real_train']
        u0_imag = dict['u0_imag_train']

        # Create a dummy class label (all zeros)
        cls = np.array(np.zeros((1,)), dtype=np.int64)

        # Normalize the velocity and convert arrays to PyTorch tensors, adding batch dimensions
        selected_v = normalizer_vel(v)
        selected_v = th.tensor(selected_v, dtype=th.float32).unsqueeze(0).to(device=device)
        x_coords = th.tensor(x_coords, dtype=th.float32).unsqueeze(0).to(device=device)
        z_coords = th.tensor(z_coords, dtype=th.float32).unsqueeze(0).to(device=device)
        sx_coords = th.tensor(sx_coords, dtype=th.float32).unsqueeze(0).to(device=device)
        cls = th.tensor(cls, dtype=th.int64).to(device=device)

        # Convert wavefield data to tensors
        u0_real = th.tensor(u0_real, dtype=th.float32).to(device=device)
        u0_imag = th.tensor(u0_imag, dtype=th.float32).to(device=device)
        
        # Additional model kwargs can be defined if necessary (currently empty)
        model_kwargs = {}

        # ------------------ Sampling ------------------
        # Generate samples using the diffusion model's sampling function.
        # This function returns:
        #   - sample: the final generated latent sample
        #   - pred_all: intermediate predictions from different timesteps
        #   - pred_xstart: the predicted x_start values (denoised output)
        #   - pde_loss_before: PDE loss values computed before diffusion correction
        #   - pde_loss_after: PDE loss values computed after diffusion correction
        sample, pred_all, pred_xstart, pde_loss_before, pde_loss_after = sample_fn(
            model, AE.decode, pinn, params_shapes, 
            x_coords, z_coords, sx_coords, omega, selected_v, u0_real, u0_imag, cls,
            (args.batch_size, args.in_channels, args.latent_dim),
            scale_factor=args.scale_factor,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        # Stack the intermediate predictions for easier handling
        pred_all = th.stack(pred_all)
        pred_xstart = th.stack(pred_xstart)

        # ------------------ Saving Results ------------------
        # Save the output results in MATLAB .mat format. The saved data includes:
        #   - predict: the final predicted latent sample
        #   - pred_all: intermediate predictions during diffusion sampling
        #   - pred_xstart: the denoised predictions
        #   - pde_loss_before: PDE loss values before applying diffusion guidance
        #   - pde_loss_after: PDE loss values after applying diffusion guidance
        if not args.use_ddim:
            sio.savemat(f'{dir_output}{fn}_out.mat', 
                {'predict': sample.squeeze().cpu().numpy(), 
                 'pred_all': pred_all.squeeze().cpu().numpy(), 
                 'pred_xstart': pred_xstart.squeeze().cpu().numpy(), 
                 'pde_loss_before': np.array(pde_loss_before, dtype=np.float32),
                 'pde_loss_after': np.array(pde_loss_after, dtype=np.float32)})
        else:
            sio.savemat(f'{dir_output}{fn}_out.mat',  
                {'predict': sample.squeeze().cpu().numpy(), 
                 'pred_all': pred_all.squeeze().cpu().numpy(), 
                 'pred_xstart': pred_xstart.squeeze().cpu().numpy(), 
                 'pde_loss_before': np.array(pde_loss_before, dtype=np.float32),
                 'pde_loss_after': np.array(pde_loss_after, dtype=np.float32)})

        print(f'Test data {fn}.mat sampling done')

    logger.log("sampling complete")

def create_argparser():
    """
    Creates an argument parser with default parameters for latent diffusion sampling.

    Returns:
    - parser: An argparse.ArgumentParser object with added default parameters.
    """
    # Define a dictionary of default parameters
    defaults = dict(
        latent_dim=1590,             # Dimensionality of the latent space
        clip_denoised=True,          # Whether to clip denoised samples
        use_ddim=True,               # Flag to determine which sampling method to use (DDIM or DDPM)
        batch_size=1,                # Batch size for sampling
        diff_path='../../trained_model/diff_trained.pt',  # Path prefix for the trained diffusion model checkpoint
        ae_path='../../trained_model/ae_trained.pth',   # Path prefix for the trained autoencoder checkpoint
        input_noise_factor=0.0,      # Noise factor for input during autoencoder processing
        latent_noise_factor=0.0,     # Noise factor for the latent representation during autoencoder processing
        freq=5,                      # Frequency used for angular frequency calculation (omega)
        scale_factor=0.01,          # Scale factor for sampling
    )
    # Update the defaults with parameters from the diffusion model configuration
    defaults.update(model_and_diffusion_defaults())
    # Create the argument parser
    parser = argparse.ArgumentParser()
    # Automatically add default parameters to the parser
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
