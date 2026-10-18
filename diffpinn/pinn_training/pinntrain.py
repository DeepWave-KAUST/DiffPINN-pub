# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Description:
# This script performs pinn training 
# It include:
# 1. Loads a pre-trained model checkpoint (from meta-training).
# 2. Loads training data.
# 3. Fine-tunes the model on the training set.
# 4. Saves heckpoints.
# -----------------------------------------------------------------------------

import  os
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  argparse
from torch.autograd import Variable
from pinnmodel import PINN
import  scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import math
from losses import PINNLoss, RegLoss
from utils import PositionalEncod, calculate_grad

# Paths to meta-trained checkpoint, model checkpoint, output directories, etc.
dir_checkpoints = './checkpoints/'
dir_meta = '../../trained_model/meta_trained.pth'
dir_pinntrain = '../../dataset/pinntrain/'
dir_output = './output/'
os.makedirs(dir_output, exist_ok=True)
os.makedirs(dir_checkpoints, exist_ok=True)

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)

    # Define the training device
    device = torch.device('cuda')

    # Define input/output dimensions for the model
    input_dim = 3     # (x, z, sx)
    output_dim = 2    # (real part, imaginary part of the wavefield)

    # A list specifying the number of neurons in each layer.
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]
    Net = PINN(neurons).to(device)

    tmp = filter(lambda x: x.requires_grad, Net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    omega = 2*math.pi*args.omega

    # Set up optimizer and learning rate scheduler for training
    optim = torch.optim.AdamW(Net.parameters(), weight_decay = 4e-5, lr=args.lr)
    milestones = [2000, 4000, 6000, 8000, 12000]
    gamma = 0.6
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)

    # Define loss functions
    criterion_pde = PINNLoss()
    criterion_reg = RegLoss()
    criterion_mae = nn.L1Loss()

    for imodel in range(args.data_num):
        print(f'=========== Model {imodel + 1} Training start ============')
        # Set up TensorBoard writer for logging fine-tuning and testing results
        writer = SummaryWriter(log_dir=f'runs/pinntrain{imodel+1}/')
        # train and test data set
        data = sio.loadmat(f'{dir_pinntrain}pinntrain{imodel+1}.mat')
        x = data['x_train']
        z = data['z_train']
        sx = data['sx_train']
        m = data['m_train']
        m0 = data['m0_train']
        u0_real = data['u0_real_train']
        u0_imag = data['u0_imag_train']
        x_test = data['x_test']
        sx_test = data['sx_test']
        z_test = data['z_test']
        du_real = data['du_real_test']
        du_imag = data['du_imag_test']

        # numpy-->cuda for train data
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(device)
        sx = torch.tensor(sx, dtype=torch.float32).to(device)
        m, m0 = torch.tensor(m, dtype=torch.float32).to(device), \
            torch.tensor(m0, dtype=torch.float32).to(device)
        u0_real, u0_imag = torch.tensor(u0_real, dtype=torch.float32).to(device), \
                       torch.tensor(u0_imag, dtype=torch.float32).to(device)

        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        z_test = torch.tensor(z_test, dtype=torch.float32).to(device)
        sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
        du_real, du_imag = torch.tensor(du_real, dtype=torch.float32).to(device), \
                           torch.tensor(du_imag, dtype=torch.float32).to(device)

        input = torch.cat([x,z,sx],-1)
        input_test = torch.cat([x_test,z_test,sx_test],-1)

        if args.use_meta:
            Net.load_state_dict(torch.load(dir_meta, map_location=device))
            print(f'load model from meta_trained model {dir_meta}')
        else:
            print(f'model start from random initialization')

        step = 0
        while True:
            Net.train()
            # optimize theta parameters
            optim.zero_grad()
  
            du_real_pred, du_imag_pred = Net(input)

            du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real_pred, du_imag_pred)

            loss_pde = criterion_pde(x, z, sx, omega, m, m0,  
                    u0_real, u0_imag, du_real_pred, du_imag_pred, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

            loss_reg = criterion_reg(x, z, sx, omega, m0, du_real_pred, du_imag_pred)

            du_real_pred_test, du_imag_pred_test = Net(input_test)

            loss_supervised = criterion_mae(du_real_pred_test, du_real) + criterion_mae(du_imag_pred_test, du_imag)

            lamda = loss_lamda(step)

            loss = 0.1*(loss_pde + loss_reg) + lamda*loss_supervised
 
            loss.backward()

            optim.step()

            step +=1

            if step > args.epoch:
                break

        with torch.no_grad():
             du_real_pred_test, du_imag_pred_test = Net(input_test)
             Net.eval()
             du_real_pred, du_imag_pred = Net(input_test)
             accs_real = (torch.pow((du_real-du_real_pred),2)).mean().item()
             accs_imag = (torch.pow((du_imag-du_imag_pred),2)).mean().item()

        print(f'Total step: {step} Accs for real: {accs_real}')
        print(f'Total step: {step} Accs for imag: {accs_imag}')
        print(f'Total step: {step} PDE loss: {loss_pde.item()}')
        print(f'Total step: {step} Supervised loss: {lamda*loss_supervised.item()}')
        sio.savemat(f'{dir_output}pinntrain{imodel+1}.mat', 
                        {'du_real': du_real_pred.cpu().numpy(), 'accs_real': accs_real,
                         'du_imag': du_imag_pred.cpu().numpy(), 'accs_imag': accs_imag})
        torch.save(Net.state_dict(), f'{dir_checkpoints}pinntrain{imodel+1}.pth')
        print(f'==========================================================')

def loss_lamda(step, step0=3000, scale=1e-3):
    lamda = 5 / (1 + math.exp(scale * (step - step0)))
    return lamda

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=15000)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1.5e-3)
    argparser.add_argument('--omega', type=float, help='frequecy', default=5)
    argparser.add_argument('--data_num', type=int, help='data_num', default=2600)
    argparser.add_argument('--use_meta', type=str, help='whether use meta-trained model', default=True)
    args = argparser.parse_args()

    main(args)
