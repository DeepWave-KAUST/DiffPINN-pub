"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description:
This script trains an autoencoder using the trained PINN parameters. It leverages a pre-defined large autoencoder
architecture from the encoder module and uses a WAE-MMD loss for regularization. The training process
employs a data prefetcher for efficient GPU usage, logs metrics via TensorBoard, and saves checkpoints periodically.
"""

import torch, os
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter  
from torch.utils.data import DataLoader  
import scipy.io as sio  
from encoder import large  # Import the large autoencoder architecture
from dataset import Basicdataset, EnlargedSampler, CUDAPrefetcher  # Custom dataset and sampler utilities
from loss import WAE_MMD_LOSS  # Custom loss function for Wasserstein Autoencoder with MMD regularization
import time

# Directories for saving checkpoints and accessing the dataset
dir_checkpoints = './checkpoints/'
dir_dataset = '../../dataset/aetrain/'

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(2222)
    torch.cuda.manual_seed_all(2222)
    np.random.seed(2222)

    # Print command-line arguments for debugging purposes
    print(args)
    # Define the device for computation (using GPU)
    device = torch.device('cuda')

    # Initialize the autoencoder model with the provided noise factors and move it to the GPU
    net = large(input_noise_factor=args.input_noise_factor, latent_noise_factor=args.latent_noise_factor).to(device)

    # Create the dataset instance from the specified directory
    dataset = Basicdataset(dir_dataset)
    # Use a custom sampler to potentially enlarge the dataset by replication or customized sampling
    sampler = EnlargedSampler(dataset, num_replicas=1, rank=0)
    # Create a DataLoader to handle batch processing of the dataset
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=sampler,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True
    )
    # Wrap the DataLoader with a CUDA prefetcher for efficient data transfer to GPU
    prefetcher = CUDAPrefetcher(train_loader)

    # Define the optimizer (AdamW) for updating the autoencoder parameters
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    # Define a learning rate scheduler to reduce the learning rate at specified milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 400, 800, 1000, 1600], gamma=0.8)

    # Initialize the custom loss function (WAE-MMD loss) with hyperparameters
    criterion = WAE_MMD_LOSS(
        args.latent_dim, 
        args.reg_weight, 
        args.reconstruction_loss_scale, 
        args.kernel_choice, 
        args.kernel_bandwidth
    )

    # Print training configuration details
    print(f'''Starting training:
        Epochs:                   {args.epoch}
        Learning rate:            {args.lr}
        Device:                   {device.type}
    ''')

    # Initialize TensorBoard writer for logging training metrics
    writer = SummaryWriter(comment=f'LR_{args.lr}_Epoch_{args.epoch}')
    # Create the directory for saving checkpoints if it does not exist
    os.makedirs(dir_checkpoints, exist_ok=True)

    print('------------ Training start ------------')

    # Begin training loop for the specified number of epochs
    for step in range(args.epoch):
        # Reset the prefetcher at the beginning of each epoch
        prefetcher.reset()
        data = prefetcher.next()

        # Initialize accumulators for losses in the current epoch
        epoch_loss = 0
        epoch_rec_loss = 0
        epoch_mmd_loss = 0
        batch_train_idx = 0

        # Process each batch in the epoch
        while data is not None:
            # Retrieve the raw input from the data batch
            label = data['raw']
            # Move the input data to the GPU
            label = label.to(device)
            batch_train_idx += 1

            # Zero the gradients before each batch update
            optimizer.zero_grad()
            
            # Forward pass: get the predicted output and latent representation from the autoencoder
            pred, latent = net(label)

            # Compute the total loss, as well as reconstruction and MMD components
            loss, recon_loss, mmd_loss = criterion(label, pred, latent)
            # Backpropagate the loss
            loss.backward()
            # Update the model parameters
            optimizer.step()

            # Get the next batch of data using the prefetcher
            data = prefetcher.next()

            # Accumulate batch losses for epoch statistics
            epoch_loss += loss
            epoch_rec_loss += recon_loss
            epoch_mmd_loss += mmd_loss

        # Compute average losses over the epoch
        epoch_loss = epoch_loss / batch_train_idx
        epoch_rec_loss = epoch_rec_loss / batch_train_idx
        epoch_mmd_loss = epoch_mmd_loss / batch_train_idx

        # Log the epoch losses to TensorBoard
        writer.add_scalar('Loss/loss_epoch', epoch_loss.item(), step)
        writer.add_scalar('Loss/rec_loss_epoch', epoch_rec_loss.item(), step)
        writer.add_scalar('Loss/mmd_loss_epoch', epoch_mmd_loss.item(), step)

        # Print loss information every 50 epochs
        if (step + 1) % 50 == 0:
            print('Epoch:', step + 1, 'loss:', epoch_loss.item())
            print('Epoch:', step + 1, 'reconstruction loss:', epoch_rec_loss.item())
            print('Epoch:', step + 1, 'mmd loss:', epoch_mmd_loss.item())

        # Save model checkpoints every 100 epochs
        if (step + 1) % 100 == 0:
            torch.save(net.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

        # Step the learning rate scheduler
        scheduler.step()

    # Close the TensorBoard writer
    writer.close()

    print('------------ Training ending ------------')

if __name__ == '__main__':
    # Create an argument parser for command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='Number of training epochs', default=1000)
    argparser.add_argument('--batch_size', type=int, help='Batch size for training', default=30)
    argparser.add_argument('--num_workers', type=int, help='Number of worker threads for data loading', default=1)
    argparser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    argparser.add_argument('--latent_dim', type=int, help='Dimensionality of the latent space', default=1590*128)
    argparser.add_argument('--reg_weight', type=float, help='Regularization weight for the MMD loss', default=100)
    argparser.add_argument('--reconstruction_loss_scale', type=float, help='Scaling factor for reconstruction loss', default=1.0)
    argparser.add_argument('--kernel_choice', type=str, help='Kernel choice for MMD loss (e.g., "imq")', default='imq')
    argparser.add_argument('--kernel_bandwidth', type=int, help='Kernel bandwidth for MMD loss', default=1)
    argparser.add_argument('--input_noise_factor', type=float, help='Noise factor to add to the input signal', default=0.0)
    argparser.add_argument('--latent_noise_factor', type=float, help='Noise factor to add to the latent representation', default=0.)

    # Parse command-line arguments and pass them to the main training function
    args = argparser.parse_args()
    main(args)
