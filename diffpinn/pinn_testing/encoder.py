import random
import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# one dimension convolutional encoder
class ODEncoder(nn.Module):
    def __init__(self, stride, kernel_size, channel_list):
        super(ODEncoder, self).__init__()
        self.channel_list = channel_list
        self.stride = stride
        self.kernel_size = kernel_size

        self.encoder = nn.ModuleList()
        channel_list = [channel_list[0]] + channel_list
        layer_num = len(channel_list) - 1
        layer = nn.Conv1d(1, channel_list[0], kernel_size, stride = 1, padding=(kernel_size - 1) // 2)
        self.encoder.append(layer)

        for index in range(layer_num):
            if_last = False
            if index == layer_num - 1:
                if_last = True
            layer = self.build_layer(if_last, kernel_size, stride, channel_list[index], channel_list[index+1])
            self.encoder.append(layer)

        self.res_block = nn.Sequential(
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            nn.Tanh(),
        )

    def build_layer(self, if_last, kernel_size, stride, inp_channel, out_channel):
        # first: if is the first layer of encoder
        layer = nn.Sequential(
            nn.Conv1d(inp_channel, out_channel, kernel_size, stride = 1, padding=(kernel_size - 1) // 2),
            nn.GroupNorm(32, out_channel),
            nn.GELU(),
            nn.Conv1d(out_channel, out_channel, kernel_size, stride = stride, padding = (kernel_size - stride) // 2),
            nn.GroupNorm(32, out_channel),
            nn.Identity() if if_last else nn.GELU(),
        )
        return layer

    def forward(self, x, **kwargs):
        for i, module in enumerate(self.encoder):
            x = module(x)
        # x = self.res_block(x)
        return x

class ODDecoder(nn.Module):
    def __init__(self, stride, kernel_size, channel_list):
        super(ODDecoder, self).__init__()
        self.channel_list = channel_list
        self.stride = stride
        self.kernel_size = kernel_size

        channel_list = channel_list + [32]
        layer_num = len(channel_list) - 1
        self.decoder = nn.ModuleList()
        for index in range(layer_num):
            layer = self.build_layer(kernel_size, stride, channel_list[index], channel_list[index+1])
            self.decoder.append(layer)

        layer = nn.Conv1d(channel_list[-1], 1, 3, stride = 1, padding=1)
        self.decoder.append(layer)

        self.res_block = nn.Sequential(
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            nn.GELU(),
        )

    def build_layer(self, kernel_size, stride, inp_channel, out_channel):
        layer = nn.Sequential(
            nn.ConvTranspose1d(inp_channel, inp_channel, kernel_size, stride=stride, padding=(kernel_size - stride) // 2),
            nn.GroupNorm(32, inp_channel),
            nn.GELU(),
            nn.Conv1d(inp_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.GroupNorm(32, out_channel),
            nn.GELU(),
        )
        return layer

    def forward(self, x, **kwargs):
        # x = self.res_block(x)
        for i, module in enumerate(self.decoder):
            x = module(x)
        return x

class ODEncoder2Decoder(nn.Module):
    def __init__(self, kernel_size=3, stride=3, input_noise_factor=0.001, 
                latent_noise_factor=0.1, enc_channel_list=None, dec_channel_list=None):
        super(ODEncoder2Decoder, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        # default augment for debug
        if enc_channel_list is None:
            enc_channel_list = [2, 2, 2, 2]
        if dec_channel_list is None:
            dec_channel_list = [2, 64, 64, 8]

        self.encoder = ODEncoder(stride, kernel_size, enc_channel_list)
        self.decoder = ODDecoder(stride, kernel_size, dec_channel_list)

        self.padder_size = stride ** len(enc_channel_list)

    def encode(self, x, **kwargs):
        _, _, h = x.shape
        x = self.check_inp_size(x)
        x = self.add_noise(x, self.input_noise_factor)
        x = self.encoder(x, **kwargs)
        return x, h

    def decode(self, x, h, **kwargs):
        x = self.add_noise(x, self.latent_noise_factor)
        x = torch.clamp(x, -1, 1)
        x = self.decoder(x, **kwargs)
        return x[:, :, :h]

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        noise_factor_rand = random.uniform(0, noise_factor)

        return torch.randn_like(x) * noise_factor_rand + x * (1 - noise_factor_rand)

    def forward(self, x, **kwargs):
        latent, h = self.encode(x)
        x = self.decode(latent, h)
        return x, latent

    def check_inp_size(self, x):
        _, _, h = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_h), mode='replicate')
        return x

class small(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        stride = 3
        kernel_size = 3
        enc_channel_list = [2, 2, 2, 2]
        dec_channel_list = [2, 64, 64, 8]
        super(small, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)

class medium(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        stride = 5
        kernel_size = 5
        enc_channel_list = [4, 4, 4, 4]
        dec_channel_list = [4, 256, 256, 8]
        super(medium, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)

class large(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        stride = 3
        kernel_size = 5
        enc_channel_list = [64, 128, 128, 128]
        dec_channel_list = [128, 512, 512, 64]
        super(large, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)

class big(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        stride = 5
        kernel_size = 5
        enc_channel_list = [32, 64, 64, 128]
        dec_channel_list = [128, 256, 256, 32]
        super(large, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)

if __name__ == '__main__':
    model = small(990721, 0.1, 0.1)
    random_tensor = torch.rand(3, 1, 990721)
    x = model(random_tensor)
    print(x.shape)