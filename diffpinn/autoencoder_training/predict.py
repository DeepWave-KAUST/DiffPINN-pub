"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description:
This script performs inference using a trained autoencoder to obtain the latent representation of PINN parameters. 
It first loads a trained autoencoder model to generate latent parameters from trained PINN parameters.
The latent parameters are then reshaped and loaded into the PINN to generate the real and imaginary
components of the wavefield. The script saves the latent representations, predicted wavefields, and evaluation
metrics (loss and accuracy) in MATLAB (.mat) files for further analysis.
"""

import torch, os
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
from encoder import ODEncoder2Decoder, medium, large  # Import various autoencoder architectures
from pinnmodel import PINN  # Import the PINN model

# Define directories for datasets, latent vectors, wavefield outputs, and the pre-trained autoencoder checkpoint
dir_dataset = '../dataset/autoencoder_dataset/'
dir_pinn_testdata = '../dataset/pinn_dataset/test/'
dir_latent = './latent_vector/'
dir_wavefield_output = './wavefield_output/'
dir_load = './checkpoints/CP_epoch2000.pth'

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    device = torch.device('cuda')

    # Load the pre-trained autoencoder model (large variant) with specified noise factors
    net = large(input_noise_factor=args.input_noise_factor, latent_noise_factor=args.latent_noise_factor).to(device)
    net.load_state_dict(torch.load(dir_load, map_location=device))
    print(f'Model loaded')

    # Define the PINN architecture: input_dim and output_dim along with hidden layer sizes
    input_dim = 3
    output_dim = 2
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]
    pinn = PINN(neurons).to(device)
    # Store the shapes of the PINN parameters for later reshaping
    params_shapes = [p.size() for p in pinn.parameters()]

    # Use Mean Squared Error loss as the evaluation criterion for autoencoder predictions
    criterion = nn.MSELoss()

    # Create output directories if they do not exist
    os.makedirs(dir_latent, exist_ok=True)
    os.makedirs(dir_wavefield_output, exist_ok=True)

    print('------------ Testing start ------------')

    # Arrays to accumulate loss and accuracy metrics over the test data
    loss_all = np.zeros((args.data_num,))
    accs_real_all = np.zeros((args.data_num,))
    accs_imag_all = np.zeros((args.data_num,))
    net.eval()  # Set the autoencoder model to evaluation mode

    data_id = 0
    # Loop through each test data sample (imodel indexes the dataset)
    for imodel in range(args.data_num):
        # Load input parameters from MATLAB file for the autoencoder
        dict = sio.loadmat(f'{dir_dataset}/data{imodel+1}.mat')
        label = dict['params']
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        label = label.to(device)

        # Inference with autoencoder: obtain predicted parameters and latent representation
        with torch.no_grad():
            pred_params, latent = net(label)
            loss = criterion(pred_params, label)

        # Record the loss for the current sample
        loss_all[data_id] = loss.item()
        # Save the latent representation to a MATLAB file for further analysis
        sio.savemat(f'{dir_latent}CurveVel_A_{imodel+1}.mat', {'latent': latent.cpu().squeeze().numpy()})

        # ------------------ Load PINN Test Data ------------------
        # Load wavefield test data corresponding to the current sample
        data = sio.loadmat(f'{dir_pinn_testdata}CurveVel_A_{imodel}.mat')
        x_test = data['x_test']
        sx_test = data['sx_test']
        z_test = data['z_test']
        du_real = data['du_real_test']
        du_imag = data['du_imag_test']

        # Convert test data arrays to torch tensors and move to GPU
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        z_test = torch.tensor(z_test, dtype=torch.float32).to(device)
        sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
        du_real = torch.tensor(du_real, dtype=torch.float32).to(device)
        du_imag = torch.tensor(du_imag, dtype=torch.float32).to(device)

        # Concatenate spatial coordinates into a single tensor for the PINN input
        input_test = torch.cat([x_test, z_test, sx_test], -1)

        # ------------------ Reshape Predicted Parameters ------------------
        # Remove extra dimensions from predicted parameters
        pred_params = pred_params.squeeze()
        params_reshape = []
        offset = 0
        # Reshape the flattened predicted parameters back to the shapes of the PINN parameters
        for shape in params_shapes:
            numel = torch.prod(torch.tensor(shape)).item()
            param = pred_params[offset:offset+numel].view(shape)
            params_reshape.append(param)
            offset += numel

        # ------------------ Update PINN Parameters ------------------
        with torch.no_grad():
            # Copy the reshaped parameters into the corresponding PINN model parameters
            for p, new_p in zip(pinn.parameters(), params_reshape):
                p.copy_(new_p)

            pinn.eval()  # Set the PINN to evaluation mode
            # Perform inference using the PINN with the test input
            du_real_pred, du_imag_pred = pinn(input_test)
            # Calculate mean squared error between predicted and ground truth wavefield data
            accs_real = (torch.pow((du_real - du_real_pred), 2)).mean().item()
            accs_imag = (torch.pow((du_imag - du_imag_pred), 2)).mean().item()

        # Record the accuracies (errors) for the current sample
        accs_real_all[data_id] = accs_real
        accs_imag_all[data_id] = accs_imag

        # Save the predicted wavefield (both real and imaginary parts) to a MATLAB file
        sio.savemat(f'{dir_wavefield_output}CurveVel_A_{imodel+1}.mat', 
                    {'du_real': du_real_pred.cpu().squeeze().numpy(), 
                     'du_imag': du_imag_pred.cpu().squeeze().numpy()})

        # Print the loss and accuracy metrics for the current sample
        print(f'CurveVel_A_{imodel} loss value {loss.item()}, accs_real {accs_real}, and accs_imag {accs_imag}')

        data_id += 1

    # Save the overall metrics (loss and accuracy) for all test samples to a MATLAB file
    sio.savemat('metric_all.mat', {'loss': loss_all, 'accs_real': accs_real_all, 'accs_imag': accs_imag_all})

    print('------------ Testing ending ------------')

if __name__ == '__main__':
    # Define command-line arguments for inference
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_noise_factor', type=float, help='Input noise factor', default=0.)
    argparser.add_argument('--latent_noise_factor', type=float, help='Latent noise factor', default=0.)
    argparser.add_argument('--data_num', type=int, help='Number of data samples for testing', default=6000)
    args = argparser.parse_args()

    main(args)
