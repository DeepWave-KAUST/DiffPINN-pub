import  os
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import  argparse
from torch.autograd import Variable
from pinnmodel import PINN
from encoder import large
import  scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import math
from losses import PINNLoss, RegLoss
from utils import calculate_grad

# Paths to autoencoder-trained checkpoint, testing dataset, generated weight directories, etc.
dir_ae = '../../trained_model/ae_trained.pth'
dir_pinndata = '../../dataset/pinntest/'
dir_diff = '../../generated_weight/'

def main(args, fn):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)

    device = torch.device('cuda')

    AE = large(input_noise_factor=args.input_noise_factor, latent_noise_factor=args.latent_noise_factor).to(device)
    AE.load_state_dict(torch.load(dir_ae, map_location=device))

    input_dim = 3
    output_dim = 2
    neurons = [input_dim, 256, 256, 128, 128, 64, 64, output_dim]
    net = PINN(neurons).to(device)

    omega = 2*math.pi*args.omega

    dir_output = f'./output/diffpinn/{fn}/'
    dir_checkpoints = f'./checkpoints/diffpinn/{fn}/'
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_checkpoints, exist_ok=True)

    print(f'------------ Testing for data {fn} ------------')
    # train data set
    data = sio.loadmat(f'{dir_pinndata}{fn}.mat')
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
    du_real_test = data['du_real_test']
    du_imag_test = data['du_imag_test']

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
    du_real_test, du_imag_test = torch.tensor(du_real_test, dtype=torch.float32).to(device), \
                       torch.tensor(du_imag_test, dtype=torch.float32).to(device)

    input = torch.cat([x,z,sx],-1)
    input_test = torch.cat([x_test,z_test,sx_test],-1)

    print('------------ Parameters generater, reshape, and load ------------')
    dict = sio.loadmat(f'{dir_diff}{fn}_weight.mat')
    sample_latent = dict['predict']
    print(sample_latent.shape)

    sample_latent = torch.tensor(sample_latent, dtype=torch.float32).unsqueeze(0).to(device)
    params_shapes = [p.size() for p in net.parameters()]

    print(sample_latent.shape)

    with torch.no_grad():
        pred_params = AE.decode(sample_latent, 128770)
    print(pred_params.shape)
    pred_params = pred_params.squeeze()
    params_reshape = []
    offset = 0
    for shape in params_shapes:
        numel = torch.prod(torch.tensor(shape)).item()
        param = pred_params[offset:offset+numel].view(shape)
        params_reshape.append(param)
        offset += numel

    with torch.no_grad():
        for p, new_p in zip(net.parameters(), params_reshape):
            p.copy_(new_p)

    print('------------ Parameters load done ------------')
    with torch.no_grad():
        du_real_pred, du_imag_pred = net(input_test)
        accs_real = (torch.pow((du_real_test - du_real_pred),2)).mean().item()
        accs_imag = (torch.pow((du_imag_test - du_imag_pred),2)).mean().item()
    print(f'Test {fn}.mat start from ini accs_real {accs_real} and accs_imag {accs_imag}')
    sio.savemat(f'{dir_output}pred_fromini.mat', 
                {'du_real_pred': du_real_pred.cpu().numpy(), 'accs_real': accs_real, 
                 'du_imag_pred': du_imag_pred.cpu().numpy(), 'accs_imag': accs_imag})

    print('------------ PINN Training start ------------')

    optimizer = torch.optim.AdamW(net.parameters(), weight_decay = 4e-5, lr=args.lr)
    milestones = [2000, 4000, 6000, 8000]
    gamma = 0.6
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    print('milestones:',milestones)
    print('gamma:',gamma)

    writer = SummaryWriter(log_dir=f'runs/diffpinn/{fn}/')

    criterion_pde = PINNLoss()
    criterion_reg = RegLoss()

    for step in range(args.epoch):
        net.train()
        optimizer.zero_grad()

        du_real_pred, du_imag_pred = net(input)

        du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real_pred, du_imag_pred)

        loss_pde = criterion_pde(x, z, sx, omega, m, m0,  
                    u0_real, u0_imag, du_real_pred, du_imag_pred, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

        loss_reg = criterion_reg(x, z, sx, omega, m0, du_real_pred, du_imag_pred)

        loss = 0.1*(loss_pde + loss_reg)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            net.eval()
            du_real_pred, du_imag_pred = net(input_test)
            accs_real = (torch.pow((du_real_test - du_real_pred),2)).mean().item()
            accs_imag = (torch.pow((du_imag_test - du_imag_pred),2)).mean().item()

        writer.add_scalar('loss_pde', loss_pde.item(), step)
        writer.add_scalar('accs_real', accs_real, step)
        writer.add_scalar('accs_imag', accs_imag, step)

        if (step + 1) % 100 == 0:
            print(f'step: {step + 1} Training loss: {loss.item()}')
            print(f'step: {step + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {step + 1} Training REG loss: {loss_reg.item()}')

            sio.savemat(f'{dir_output}pred{step+1}.mat',
                {'du_real_pred': du_real_pred.cpu().numpy(), 'accs_real': accs_real,
                 'du_imag_pred': du_imag_pred.cpu().numpy(), 'accs_imag': accs_imag})

            print(f'Test accs real: {accs_real}')
            print(f'Test accs imag: {accs_imag}')

        if (step + 1) % 500 == 0:
            torch.save(net.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

        scheduler.step()

    print('---------------------------------------------------------')
    print('-------------------- Ending Training --------------------')
    print('---------------------------------------------------------')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1.5e-3)
    argparser.add_argument('--omega', type=float, help='frequecy', default=5)
    argparser.add_argument('--input_noise_factor', type=float, help='input_noise_factor', default=0.)
    argparser.add_argument('--latent_noise_factor', type=float, help='latent_noise_factor', default=0.)

    args = argparser.parse_args()

    data_list = ['CurveVelA', 'FlatFaultA', 'Layer']

    for fn in data_list:
        main(args, fn)

