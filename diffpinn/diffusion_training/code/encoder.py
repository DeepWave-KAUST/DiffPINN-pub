"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com

Description:
This module defines a one-dimensional convolutional autoencoder architecture.
It contains an encoder and a decoder with residual blocks (ResBlock) and configurable
layer channels and kernel sizes. Three variants (small, medium, large) are provided
with different capacities. The autoencoder is designed for processing 1D signals.
"""

import random
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------- Residual Block ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes a Residual Block that adds a residual connection over a convolutional block.

        Arguments:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels for the inner convolution.
                        Note: The block's output is added to the input, so the output of the conv block
                        must have the same shape as the input.
        """
        nn.Module.__init__(self)
        # Define a sequential convolutional block with GELU activations
        self.conv_block = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """
        Forward pass for the residual block.

        Arguments:
        - x: Input tensor.

        Returns:
        - Output tensor: x added to the output of the convolutional block.
        """
        return x + self.conv_block(x)

# ---------------------------- One-Dimensional Encoder ----------------------------
class ODEncoder(nn.Module):
    def __init__(self, stride, kernel_size, channel_list):
        """
        One-dimensional convolutional encoder.

        Arguments:
        - stride: Stride used for downsampling in convolutional layers.
        - kernel_size: Size of the convolutional kernels.
        - channel_list: A list defining the number of channels for each layer.
                        The first element is used as the initial number of channels.
        """
        super(ODEncoder, self).__init__()
        self.channel_list = channel_list
        self.stride = stride
        self.kernel_size = kernel_size

        # Build encoder layers as a ModuleList
        self.encoder = nn.ModuleList()
        # Prepend the first channel from channel_list to itself for initial convolution
        channel_list = [channel_list[0]] + channel_list
        layer_num = len(channel_list) - 1
        
        # First layer: convert input from 1 channel to channel_list[0]
        layer = nn.Conv1d(1, channel_list[0], kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.encoder.append(layer)

        # Build subsequent layers based on channel_list
        for index in range(layer_num):
            # Check if this is the last layer in the encoder
            if_last = (index == layer_num - 1)
            layer = self.build_layer(if_last, kernel_size, stride, channel_list[index], channel_list[index+1])
            self.encoder.append(layer)

        # Define a series of residual blocks followed by a Tanh activation
        self.res_block = nn.Sequential(
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            nn.Tanh(),
        )

    def build_layer(self, if_last, kernel_size, stride, inp_channel, out_channel):
        """
        Constructs a convolutional layer block for the encoder.

        Arguments:
        - if_last: Boolean indicating if this is the last layer of the encoder.
        - kernel_size: Size of the convolution kernel.
        - stride: Stride used for downsampling.
        - inp_channel: Number of input channels.
        - out_channel: Number of output channels.

        Returns:
        - A sequential module representing the layer block.
        """
        layer = nn.Sequential(
            # First convolution: maintains spatial dimensions
            nn.Conv1d(inp_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.GroupNorm(32, out_channel),
            nn.GELU(),
            # Second convolution: applies stride for downsampling
            nn.Conv1d(out_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size - stride) // 2),
            nn.GroupNorm(32, out_channel),
            # Use Identity if it's the last layer, otherwise apply GELU activation
            nn.Identity() if if_last else nn.GELU(),
        )
        return layer

    def forward(self, x, **kwargs):
        """
        Forward pass for the encoder.

        Arguments:
        - x: Input tensor with shape (batch_size, 1, length).
        - kwargs: Additional keyword arguments (if any).

        Returns:
        - Encoded feature tensor.
        """
        # Sequentially pass through each layer in the encoder
        for module in self.encoder:
            x = module(x)
        # Optionally, a residual block can be applied (currently commented out)
        # x = self.res_block(x)
        return x

# ---------------------------- One-Dimensional Decoder ----------------------------
class ODDecoder(nn.Module):
    def __init__(self, stride, kernel_size, channel_list):
        """
        One-dimensional convolutional decoder.

        Arguments:
        - stride: Stride used for upsampling in transposed convolution layers.
        - kernel_size: Size of the convolutional kernels.
        - channel_list: A list defining the number of channels for each layer.
        """
        super(ODDecoder, self).__init__()
        self.channel_list = channel_list
        self.stride = stride
        self.kernel_size = kernel_size

        # Extend channel list by appending an additional channel dimension (e.g., 32)
        channel_list = channel_list + [32]
        layer_num = len(channel_list) - 1
        # Build decoder layers as a ModuleList
        self.decoder = nn.ModuleList()
        for index in range(layer_num):
            layer = self.build_layer(kernel_size, stride, channel_list[index], channel_list[index+1])
            self.decoder.append(layer)

        # Final layer to map back to a single channel output
        layer = nn.Conv1d(channel_list[-1], 1, 3, stride=1, padding=1)
        self.decoder.append(layer)

        # Define a series of residual blocks with GELU activations (optional post-processing)
        self.res_block = nn.Sequential(
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[0], channel_list[0] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            ResBlock(channel_list[-1], channel_list[-1] // 2),
            nn.GELU(),
        )

    def build_layer(self, kernel_size, stride, inp_channel, out_channel):
        """
        Constructs a decoder layer block using transposed convolutions.

        Arguments:
        - kernel_size: Size of the convolution kernel.
        - stride: Stride used for upsampling.
        - inp_channel: Number of input channels.
        - out_channel: Number of output channels.

        Returns:
        - A sequential module representing the decoder layer.
        """
        layer = nn.Sequential(
            # Transposed convolution to upsample the feature map
            nn.ConvTranspose1d(inp_channel, inp_channel, kernel_size, stride=stride, padding=(kernel_size - stride) // 2),
            nn.GroupNorm(32, inp_channel),
            nn.GELU(),
            # Standard convolution to adjust the channel dimensions
            nn.Conv1d(inp_channel, out_channel, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.GroupNorm(32, out_channel),
            nn.GELU(),
        )
        return layer

    def forward(self, x, **kwargs):
        """
        Forward pass for the decoder.

        Arguments:
        - x: Input tensor from the latent space.
        - kwargs: Additional keyword arguments (if any).

        Returns:
        - Reconstructed tensor with the original input dimensions.
        """
        # Optionally, apply residual blocks before decoding (currently commented out)
        # x = self.res_block(x)
        # Sequentially pass through each decoder layer
        for module in self.decoder:
            x = module(x)
        return x

# ---------------------------- Encoder-Decoder Wrapper ----------------------------
class ODEncoder2Decoder(nn.Module):
    def __init__(self, kernel_size=3, stride=3, input_noise_factor=0.001, 
                 latent_noise_factor=0.1, enc_channel_list=None, dec_channel_list=None):
        """
        Combines the one-dimensional encoder and decoder to form an autoencoder.

        Arguments:
        - kernel_size: Convolution kernel size.
        - stride: Stride used in the encoder/decoder.
        - input_noise_factor: Noise factor to add to the input signal.
        - latent_noise_factor: Noise factor to add to the latent representation.
        - enc_channel_list: List of channels for the encoder layers.
        - dec_channel_list: List of channels for the decoder layers.
        """
        super(ODEncoder2Decoder, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor

        # Set default channel lists if not provided
        if enc_channel_list is None:
            enc_channel_list = [2, 2, 2, 2]
        if dec_channel_list is None:
            dec_channel_list = [2, 64, 64, 8]

        # Instantiate the encoder and decoder modules
        self.encoder = ODEncoder(stride, kernel_size, enc_channel_list)
        self.decoder = ODDecoder(stride, kernel_size, dec_channel_list)

        # Calculate padder size required to make input dimensions compatible
        self.padder_size = stride ** len(enc_channel_list)

    def encode(self, x, **kwargs):
        """
        Encodes the input signal into a latent representation.

        Arguments:
        - x: Input tensor with shape (batch_size, 1, length).

        Returns:
        - A tuple (latent representation, original length h).
        """
        # Get original signal length
        _, _, h = x.shape
        # Ensure input size is compatible with network by padding if necessary
        x = self.check_inp_size(x)
        # Add noise to the input signal
        x = self.add_noise(x, self.input_noise_factor)
        # Pass through the encoder
        x = self.encoder(x, **kwargs)
        return x, h

    def decode(self, x, h, **kwargs):
        """
        Decodes the latent representation back to the original signal.

        Arguments:
        - x: Latent representation.
        - h: Original signal length before encoding.
        - kwargs: Additional keyword arguments (if any).

        Returns:
        - Reconstructed signal truncated to original length h.
        """
        # Add noise to the latent representation
        x = self.add_noise(x, self.latent_noise_factor)
        # Clamp the values to ensure they are in a reasonable range
        x = torch.clamp(x, -1, 1)
        # Pass through the decoder
        x = self.decoder(x, **kwargs)
        # Truncate the output to the original length
        return x[:, :, :h]

    def add_noise(self, x, noise_factor):
        """
        Adds noise to the input tensor.

        Arguments:
        - x: Input tensor.
        - noise_factor: Maximum noise level (or a tuple specifying a range).

        Returns:
        - Tensor with added noise.
        """
        # If noise_factor is not a float, assume it is a range [min, max]
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])
        # Randomly scale the noise factor
        noise_factor_rand = random.uniform(0, noise_factor)
        # Add Gaussian noise to x
        return torch.randn_like(x) * noise_factor_rand + x * (1 - noise_factor_rand)

    def forward(self, x, **kwargs):
        """
        Full forward pass through the autoencoder.

        Arguments:
        - x: Input tensor.
        - kwargs: Additional keyword arguments (if any).

        Returns:
        - A tuple (reconstructed signal, latent representation).
        """
        latent, h = self.encode(x)
        x = self.decode(latent, h)
        return x, latent

    def check_inp_size(self, x):
        """
        Pads the input tensor so that its length is a multiple of the required padder_size.

        Arguments:
        - x: Input tensor with shape (batch_size, 1, length).

        Returns:
        - Padded tensor.
        """
        _, _, h = x.size()
        # Calculate the required padding to make h a multiple of padder_size
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        # Pad along the last dimension using 'replicate' mode to avoid artifacts
        x = F.pad(x, (0, mod_pad_h), mode='replicate')
        return x

# ---------------------------- Model Variants ----------------------------
class small(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        """
        Small autoencoder variant with lower channel capacities.

        Arguments:
        - input_noise_factor: Noise factor for input.
        - latent_noise_factor: Noise factor for latent space.
        """
        stride = 3
        kernel_size = 3
        enc_channel_list = [2, 2, 2, 2]
        dec_channel_list = [2, 64, 64, 8]
        super(small, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, 
                                    enc_channel_list, dec_channel_list)

class medium(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        """
        Medium autoencoder variant with moderate channel capacities.

        Arguments:
        - input_noise_factor: Noise factor for input.
        - latent_noise_factor: Noise factor for latent space.
        """
        stride = 5
        kernel_size = 5
        enc_channel_list = [4, 4, 4, 4]
        dec_channel_list = [4, 256, 256, 8]
        super(medium, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, 
                                     enc_channel_list, dec_channel_list)

class large(ODEncoder2Decoder):
    def __init__(self, input_noise_factor, latent_noise_factor):
        """
        Large autoencoder variant with higher channel capacities for more complex representations.

        Arguments:
        - input_noise_factor: Noise factor for input.
        - latent_noise_factor: Noise factor for latent space.
        """
        stride = 3
        kernel_size = 5
        enc_channel_list = [64, 128, 128, 128]
        dec_channel_list = [128, 512, 512, 64]
        super(large, self).__init__(kernel_size, stride, input_noise_factor, latent_noise_factor, 
                                    enc_channel_list, dec_channel_list)

# ---------------------------- Testing ----------------------------
if __name__ == '__main__':
    # Test the autoencoder with the 'small' variant.
    # Create an instance with specified noise factors.
    model = small(990721, 0.1)
    # Generate a random input tensor with shape (batch_size, channels, length)
    random_tensor = torch.rand(3, 1, 990721)
    # Pass the random tensor through the autoencoder
    x, latent = model(random_tensor)
    # Print the shape of the reconstructed output
    print(x.shape)
