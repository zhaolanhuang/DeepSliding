# [1] A. van den Oord et al., “WaveNet: A Generative Model for Raw Audio,” Sep. 19, 2016, arXiv: arXiv:1609.03499. doi: 10.48550/arXiv.1609.03499.
# [2] T. L. Paine et al., “Fast Wavenet Generation Algorithm,” Nov. 29, 2016, arXiv: arXiv:1611.09482. doi: 10.48550/arXiv.1611.09482.


# Code Taken from https://github.com/golbin/WaveNet/blob/master/wavenet/networks.py
import torch
import torch.nn as nn



import numpy as np

class InputSizeError(Exception):
    def __init__(self, input_size, receptive_fields, output_size):

        message = 'Input size has to be larger than receptive_fields\n'
        message += 'Input size: {0}, Receptive fields size: {1}, Output size: {2}'.format(
            input_size, receptive_fields, output_size)

        super(InputSizeError, self).__init__(message)


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        # self.conv = torch.nn.Conv1d(in_channels, out_channels,
        #                             kernel_size=2, stride=1, padding=1,
        #                             bias=False)  # Fixed for WaveNet but not sure
        # ZL: not support padding now
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        # return output[:, :, :-1]
        # ZL: not support padding now, adapt
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        # print("x:", x.shape)
        output = self.dilated(x)
        # print("dilation:", self.dilated.conv.dilation)
        # print("output:", output.shape)
        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        # print("input_cut:", input_cut.shape)
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        # print("skip:", skip.shape)
        skip = skip[:, :, -skip_size:]
        # print("skip after crop:", skip.shape)
        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)


        # ZL: DISABLE, Only Support CPU now
        # if torch.cuda.device_count() > 1:
        #     block = torch.nn.DataParallel(block)

        # if torch.cuda.is_available():
        #     block.cuda()

        return block

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        :return:
        """
        
        # res_blocks = []
        # ZL: small adaption so the res block as submodules
        res_blocks = nn.ModuleList()
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)

        return res_blocks

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next inputcalc_output_size
            output, skip = res_block(output, skip_size)
            # print("output:", skip.shape)
            # print("skip:", skip.shape)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class DensNet(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DensNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        output = self.softmax(output)

        return output


class WaveNet(torch.nn.Module):
    DEFAULT_INPUT_SHAPE = (1, 256, 10000) # (C, T)
    DEFAULT_SLIDING_STEP_SIZE = 1
    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, layer_size=10, stack_size=5, in_channels=256, res_channels=512):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WaveNet, self).__init__()

        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels)

        self.densnet = DensNet(in_channels)

        self.output_size = self.calc_output_size(input_shape)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    #change var x to its shape, avoid error in symbolc trace.
    def calc_output_size(self, x):
        # Origin:
        # output_size = int(x.size(2)) - self.receptive_fields
        # ZL: aligned with adaption of padding
        output_size = int(x[2]) - self.receptive_fields - 1 

        self.check_input_size(x, output_size)

        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x[2]), self.receptive_fields, output_size)

    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, channels, timestep]
        :return: Tensor[batch, channels, timestep]
        """
        output = x # ZL: rearrange channels and timestep

        # output_size = self.calc_output_size(output)

        output = self.causal(output)

        skip_connections = self.res_stack(output, self.output_size)

        output = torch.sum(skip_connections, dim=0)

        output = self.densnet(output)

        return output.contiguous()


class TinyWaveNet(WaveNet):
    def __init__(self):
        super().__init__(layer_size=9, stack_size=1, in_channels=256, res_channels=32)