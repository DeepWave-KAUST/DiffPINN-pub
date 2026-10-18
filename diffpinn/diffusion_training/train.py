"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description:
This script trains a latent conditioned diffusion model to generate PINN weights.
It uses a pre-trained autoencoder to decode latent representations, which are then refined via a diffusion process.
The script handles command-line argument parsing, model and diffusion process creation, data loading, and the training loop.
"""

import argparse  
from code import logger  
from code.datasets import load_data  # Function to load the training dataset
from code.resample import create_named_schedule_sampler  # Function to create a schedule sampler for diffusion
from code.script_util import (
    model_and_diffusion_defaults,  # Function returning default parameters for the model and diffusion process
    create_model_and_diffusion,    # Function to create the model and its associated diffusion process
    args_to_dict,                  # Utility to convert argparse arguments into a dictionary
    add_dict_to_argparser,         # Utility to add dictionary items as arguments to the argparse parser
)
from code.train_util import TrainLoop  # Class that encapsulates the training loop
import torch as th  # PyTorch for tensor operations and model management
from code.encoder import large  # Import the 'large' autoencoder model from the encoder module

def main():
    # Parse command-line arguments
    args = create_argparser().parse_args()

    # Configure the logging system
    logger.configure()

    logger.log("creating model and diffusion...")

    # Create the diffusion model and its process using the combined default and user-provided parameters
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Define the device for training (using GPU)
    device = th.device('cuda')

    # Move the diffusion model to the GPU
    model.to(device)

    # Create a schedule sampler for selecting time steps during the diffusion process
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Instantiate the pre-trained autoencoder with specified noise factors and move it to the GPU
    AE = large(input_noise_factor=args.input_noise_factor, latent_noise_factor=args.latent_noise_factor).to(device)
    # Load the pre-trained autoencoder weights from the checkpoint file
    AE.load_state_dict(th.load(args.ae_path, map_location=device))

    # Freeze all parameters of the autoencoder to prevent them from being updated during training
    for param in AE.parameters():
        param.requires_grad = False

    logger.log("creating data loader...")
    # Load the training dataset with specified parameters
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # Initialize and run the training loop with the model, diffusion process, autoencoder decoder, and data
    TrainLoop(
        model=model,
        diffusion=diffusion,
        decoder=AE.decode,            # Use the autoencoder's decode function as the decoder
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()  # Start the training loop

def create_argparser():
    """
    Automatically creates an argparse.ArgumentParser with default parameters.
    This function defines a dictionary of default arguments and adds them to the parser.
    """
    # Define a dictionary with default values for the training and diffusion process parameters
    defaults = dict(
        data_dir="../../dataset/difftrain/",  # Directory of the training dataset
        ae_path='../../trained_model/ae_trained.pth',   # Path prefix for the trained autoencoder checkpoint
        schedule_sampler="uniform",              # Type of schedule sampler to use
        lr=5e-5,                                 # Learning rate for training
        weight_decay=5e-5,                       # Weight decay factor for regularization
        lr_anneal_steps=0,                       # Number of steps for learning rate annealing
        batch_size=20,                           # Batch size for training
        microbatch=-1,                           # Microbatch size (-1 disables microbatching)
        ema_rate="0.999",                        # Exponential Moving Average (EMA) decay rate
        log_interval=100,                        # Interval (in iterations) for logging training status
        save_interval=10000,                     # Interval (in iterations) for saving checkpoints
        resume_checkpoint="",                    # Path to a checkpoint to resume training from
        use_fp16=False,                          # Flag to enable FP16 (half-precision) training
        fp16_scale_growth=1e-3,                    # Growth factor for FP16 dynamic scaling
        input_noise_factor=0.0,                  # Noise factor for the input data
        latent_noise_factor=0.0,                 # Noise factor for the latent representation
    )
    # Update the defaults with additional parameters for the model and diffusion process
    defaults.update(model_and_diffusion_defaults())
    
    # Create an ArgumentParser instance
    parser = argparse.ArgumentParser()
    
    # Automatically add the default parameters to the parser
    add_dict_to_argparser(parser, defaults)
    return parser

# If this script is executed as the main program, run the main() function
if __name__ == "__main__":
    main()
