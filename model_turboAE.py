import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from numpy import arange
from numpy.random import mtrand
import math
import numpy as np


class TurboConfig:
    num_iteration = 6
    code_rate_k = 1
    dec_num_layer = 5
    dec_num_unit = 100
    dec_kernel_size = 5
    enc_num_unit = 100
    enc_num_layer = 2
    enc_kernel_size = 5
    num_iter_ft = 5
    #batch_size = 32
    block_len = 24
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
def circular_pad(tensor, pad):
    """Apply circular padding to a 1D tensor."""
    return torch.cat([tensor[:, :, -pad:], tensor, tensor[:, :, :pad]], dim=2)


class Interleaver(torch.nn.Module):
    """Handles both interleaving and de-interleaving based on a given permutation array."""
    
    def __init__(self, config):
        super(Interleaver, self).__init__()
        seed = 0
        rand_gen = mtrand.RandomState(seed)
        p_array = rand_gen.permutation(arange(config.block_len))
        self.set_parray(p_array)
        print("Array: ", p_array)

    def set_parray(self, p_array):
        """Sets permutation array and its reverse."""
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

        reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            reverse_p_array[p_array[idx]] = idx
        self.reverse_p_array = torch.LongTensor(reverse_p_array).view(len(p_array))

    def _permute(self, inputs, permutation_array):
        """Permute the given input using the provided permutation array."""
        inputs = inputs.permute(1, 0, 2)
        res = inputs[permutation_array]
        return res.permute(1, 0, 2)

    def interleave(self, inputs):
        return self._permute(inputs, self.p_array)
    
    def deinterleave(self, inputs):
        return self._permute(inputs, self.reverse_p_array)


class ModuleLambda(nn.Module):
    def __init__(self, lambd):
        super(ModuleLambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
def build_encoder_block(num_layer, in_channels, out_channels, kernel_size, activation='elu'):
    layers = []
    layers.append(ModuleLambda(lambda x: torch.transpose(x, 1, 2)))
   
    
    for idx in range(num_layer):
        in_ch = in_channels if idx == 0 else out_channels
        
        # Add circular padding before the convolution, experimental
        pad = kernel_size // 2
        #layers.append(ModuleLambda(lambda x: circular_pad(x, pad)))
        
            
        layers.append(nn.Conv1d(
            in_channels=in_ch, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=pad,# padding,#  -> if without circular padding
            dilation=1, 
            groups=1, 
            bias=True
        ))

        #layers.append(nn.LayerNorm([out_channels, 100])) # Experimental; for bigger kernel sizes
        layers.append(ModuleLambda(lambda x: getattr(F, activation)(x)))
        #layers.append(nn.Dropout(0.1)) # Experimental
    
    layers.append(ModuleLambda(lambda x: torch.transpose(x, 1, 2)))
    return nn.Sequential(*layers)


class ENC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        
        self.enc_linear_1 = nn.Linear(config.enc_num_unit, 1)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 1)


    def set_parallel(self):
        self.enc_cnn_1 = nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = nn.DataParallel(self.enc_cnn_2)
        self.enc_linear_1 = nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = nn.DataParallel(self.enc_linear_2)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        
        x_sys = self.enc_cnn_1(inputs)
        x_sys = F.elu(self.enc_linear_1(x_sys))
        
        x_p1 = self.enc_cnn_2(self.interleaver.interleave(inputs))
        x_p1 = F.elu(self.enc_linear_2(x_p1))

        x_tx = torch.cat([x_sys, x_p1], dim=2)
        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)

class DEC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo, self).__init__()
        
        self.config = config
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])

    
    def set_parallel(self):
        for idx in range(self.config.num_iteration):
            self.dec1_cnns[idx] = nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = nn.DataParallel(self.dec2_outputs[idx])
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        config = self.config
        received = received.type(torch.FloatTensor).to(self.this_device)
    
        # Initial processing
        r_sys = received[:, :, 0].view((bs, config.block_len, 1))
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, config.block_len, 1))
        r_par_deint = self.interleaver.deinterleave(r_par)
    
        # Initialize prior
        prior = torch.zeros((bs, config.block_len, config.num_iter_ft)).to(self.this_device)
    
        # Turbo Decoder Loop
        for idx in range(config.num_iteration - 1):
            x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
        
            x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)
        
        # Last round
        x_dec, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
    
        x_dec, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        final = torch.sigmoid(self.interleaver.deinterleave(x_plr))
    
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, cnn, linear):
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr
    
###
###

### Testing section


class DEC_CNNTurbo_testing(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo_testing, self).__init__()
        
        self.config = config
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.transform_layer = nn.Linear(config.dec_num_unit, 2)
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])

    
    def set_parallel(self):
        for idx in range(self.config.num_iteration):
            self.dec1_cnns[idx] = nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = nn.DataParallel(self.dec2_outputs[idx])
    
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        received = received.type(torch.FloatTensor).to(self.this_device)
        
        # Initialize prior
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft)).to(self.this_device)
        
        # Iterative Decoding Loop
        for idx in range(self.config.num_iteration-1):
            x_plr1 = self._turbo_decoder_step_in(received, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr1_ex = x_plr1 - prior  # Compute extrinsic information
            x_plr1_ex = self.interleaver.deinterleave(x_plr1_ex)  # Interleave the extrinsic information
            
            x_dec2, x_plr2 = self._turbo_decoder_step_out(x_plr1_ex, x_plr1_extrinsic_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            x_plr2_extrinsic = x_plr2 - x_plr1_extrinsic_int  # Compute extrinsic information
            prior = self.interleaver.interleave(x_plr2_extrinsic)  # Deinterleave the extrinsic information for the next iteration
        # -----------    
        # last round
        x_dec1, x_plr1 = self._turbo_decoder_step(received, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr1_extrinsic = x_plr1 - prior  # Compute extrinsic information
        x_plr1_extrinsic_int = self.interleaver.deinterleave(x_plr1_extrinsic)  # Interleave the extrinsic information
            
        # Decoder 2 Processing
        x_dec1_transf = self.transform_layer(x_dec1)
        x_dec2, x_plr2 = self._turbo_decoder_step(x_dec1_transf, x_plr1_extrinsic_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        out = torch.sigmoid(self.interleaver.interleave(x_plr2))
        
        return out.squeeze(dim=2)  # Return the final output of Decoder 2

    def _turbo_decoder_step_in(self, input, prior, cnn, linear):
        x_this_dec = torch.cat([input, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_out = linear(x_dec)
        return x_out
    
    def _turbo_decoder_step_out(self, input, cnn, linear):
        x_dec = cnn(input)
        x_plr = linear(x_dec)
        return x_dec, x_plr 



            
class STEBinarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        # Save inputs for backward computation
        ctx.save_for_backward(inputs)
        
        # Binarize the inputs
        outputs = torch.sign(inputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Zero out gradients where the absolute value of the input is greater than 1
        grad_output[input > 1.0] = 0
        grad_output[input < -1.0] = 0
        # Clamp the gradient values
        grad_output = torch.clamp(grad_output, -0.25, +0.25)
        
        return grad_output

### This just STEs the gradient around
# Apparently dont need this, because torch can gradient through indexing and permutations
class InterleaveFunction(Function):
    @staticmethod
    def forward(ctx, x, permutation):
        ctx.permutation = permutation
        x = x.permute(1, 0, 2)
        res = x[permutation]
        return res.permute(1, 0, 2)

    @staticmethod
    def backward(ctx, grad_output):
        inverse_permutation = torch.argsort(ctx.permutation)
        grad_output = grad_output.permute(1, 0, 2)
        grad_input = grad_output[inverse_permutation]
        return grad_input.permute(1, 0, 2), None

class LearnableInterleaver(Interleaver):
    def __init__(self, config):
        super().__init__(config)
        
    def _permute(self, inputs, permutation_array):
        return InterleaveFunction.apply(inputs, permutation_array)
    
#### This tries to create a soft permutation, which is also learnable
# sadly it doesnt work
    
class GumbelInterleaveFunction(Function):
    @staticmethod
    def forward(ctx, x, logits, temperature=1.001):
        # Sample from Gumbel(0, 1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        
        # Apply Gumbel Softmax trick
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)        
        # Store for backward pass
        ctx.save_for_backward(y)
        
        return GumbelInterleaver._apply_permutation(x, y)
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        
        grad_input = GumbelInterleaver._apply_permutation(grad_output, y)
        return grad_input, None, None

class GumbelInterleaver(nn.Module):
    def __init__(self, config, permutation=None):
        super(GumbelInterleaver, self).__init__()
        input_dim = config.block_len
        self.logits = nn.Parameter(torch.randn(input_dim, input_dim))
        
        if permutation is not None:
            self.logits.data = self.initialize_logits_from_permutation(permutation)
    
    def interleave(self, x, temperature=1.001):
        return GumbelInterleaveFunction.apply(x, self.logits, temperature)
    
    def deinterleave(self, x, temperature=1.001):
        # Use transpose of the softmax matrix for de-interleaving
        y_transpose = F.softmax(self.logits / temperature, dim=-1).t()
        return self._apply_permutation(x, y_transpose)

    @staticmethod
    def _apply_permutation(x, y):
        """Helper function to apply permutation based on y."""
        original_shape = x.shape
        x_2d = x.reshape(original_shape[0] * original_shape[2], original_shape[1])
        out_2d = torch.mm(x_2d, y.t())
        return out_2d.reshape(original_shape)
    
    @staticmethod
    def initialize_logits_from_permutation(permutation):
        """
        Initialize the logits matrix from a given permutation.
        """
        block_len = len(permutation)
        logits = -1e6 * torch.ones(block_len, block_len)  # Using a large negative value to make other entries negligible
        for idx, perm_idx in enumerate(permutation):
            logits[idx, perm_idx] = 1e6  # Setting a large positive value to the desired permutation position
        return F.softmax(logits, dim=-1)  # Convert to softmaxed version
    
    
#### Testing serial ####


class ENC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo_serial, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Updated the encoder CNN layers. Now the first CNN produces k*Fc features and the second one processes them
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.code_rate_k * 10, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k * 10, config.enc_num_unit, config.enc_kernel_size)
        
        self.enc_linear_1 = nn.Linear(config.code_rate_k * 10, 1)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 1)

    def set_parallel(self):
        self.enc_cnn_1 = nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = nn.DataParallel(self.enc_cnn_2)
        self.enc_linear_1 = nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = nn.DataParallel(self.enc_linear_2)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        
        # Output from first CNN structure
        x_sys = self.enc_cnn_1(inputs)
        x_sys_ = F.elu(self.enc_linear_1(x_sys))
        #print("xsys",x_sys.shape)
        
        # Interleave output of first CNN
        x_sys = STEBinarize.apply(x_sys)
        x_sys_interleaved = self.interleaver.interleave(x_sys)
        
        # Pass interleaved output through second CNN structure
        x_p1 = self.enc_cnn_2(x_sys_interleaved)
        
        x_p1 = F.elu(self.enc_linear_2(x_p1))
        #print("xp1", x_p1.shape)
        x_tx = torch.cat([x_sys_, x_p1], dim=2)
        
        # self.power_constraint(x_p1)
        #
        # but then -> self.enc_linear_2 = nn.Linear(config.enc_num_unit, 2)
        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)
    
    
    
class DEC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super(DEC_CNNTurbo_serial, self).__init__()
        
        self.config = config
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.transform_layer = nn.Linear(config.dec_num_unit, 2)
        self.interleaver = interleaver
        
        self.dec1_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec2_cnns = torch.nn.ModuleList([
            build_encoder_block(config.dec_num_layer, 2 + config.num_iter_ft, config.dec_num_unit, config.dec_kernel_size)
            for _ in range(config.num_iteration)
        ])
        
        self.dec1_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, config.num_iter_ft) for _ in range(config.num_iteration)
        ])
        
        self.dec2_outputs = torch.nn.ModuleList([
            torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else config.num_iter_ft)
            for idx in range(config.num_iteration)
        ])

    
    def set_parallel(self):
        for idx in range(self.config.num_iteration):
            self.dec1_cnns[idx] = nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = nn.DataParallel(self.dec2_outputs[idx])
    
    
    def forward(self, received):
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2)
        received = received.type(torch.FloatTensor).to(self.this_device)
        
        # Initialize prior
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft)).to(self.this_device)
        
        # Iterative Decoding Loop
        for idx in range(self.config.num_iteration-1):
            x_dec1, x_plr1 = self._turbo_decoder_step(received, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr1_extrinsic = x_plr1 - prior  # Compute extrinsic information
            x_plr1_extrinsic_int = self.interleaver.deinterleave(x_plr1_extrinsic)  # Interleave the extrinsic information
            
            x_dec1_transf = self.transform_layer(x_dec1)
            x_dec2, x_plr2 = self._turbo_decoder_step(x_dec1_transf, x_plr1_extrinsic_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            x_plr2_extrinsic = x_plr2 - x_plr1_extrinsic_int  # Compute extrinsic information
            prior = self.interleaver.interleave(x_plr2_extrinsic)  # Deinterleave the extrinsic information for the next iteration
        # -----------    
        # last round
        x_dec1, x_plr1 = self._turbo_decoder_step(received, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr1_extrinsic = x_plr1 - prior  # Compute extrinsic information
        x_plr1_extrinsic_int = self.interleaver.deinterleave(x_plr1_extrinsic)  # Interleave the extrinsic information
            
        # Decoder 2 Processing
        x_dec1_transf = self.transform_layer(x_dec1)
        x_dec2, x_plr2 = self._turbo_decoder_step(x_dec1_transf, x_plr1_extrinsic_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        out = torch.sigmoid(self.interleaver.interleave(x_plr2))
        
        return out.squeeze(dim=2)  # Return the final output of Decoder 2

    def _turbo_decoder_step(self, input, prior, cnn, linear):
        x_this_dec = torch.cat([input, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr

#### 2D Experiments

def circular_pad_2d(x, pad):
    pad_h, pad_w = pad
    # Circular padding for the height (sequence length)
    top_pad = x[:, :, -pad_h:, :]
    bottom_pad = x[:, :, :pad_h, :]
    x = torch.cat([top_pad, x, bottom_pad], dim=2)
    
    return x


def build_encoder_block_2d(num_layer, in_channels, out_channels, kernel_size, activation='elu'):
    layers = []
    
    def print_shape(x):
        print(x.shape)
        return x
    
    for idx in range(num_layer):
        in_ch = in_channels if idx == 0 else out_channels
        
        # Add modified padding before the convolution for 2D
        pad_h = kernel_size // 2
        pad_w = 1  # Since the width is 2, we can pad by 1 on each side
        layers.append(ModuleLambda(print_shape))
        layers.append(ModuleLambda(lambda x, pad_h=pad_h, pad_w=pad_w: circular_pad_2d(x, (pad_h, pad_w))))
        layers.append(ModuleLambda(print_shape))
        
        layers.append(nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_channels, 
            kernel_size=(kernel_size, 3),  # Adjusted kernel size to consider both sequences as spatial dimensions
            stride=1, 
            padding=(0, 1),  # We handle the width padding using our modified padding function
            dilation=1, 
            groups=1, 
            bias=True
        ))

        #norm_layer = nn.LayerNorm([out_channels, 64, 2])  # We set the height (sequence length) to a default value; it will adjust based on the input tensor
        #layers.append(ModuleLambda(lambda x, norm_layer=norm_layer: norm_layer(x)))
        layers.append(ModuleLambda(lambda x: getattr(F, activation)(x)))
        layers.append(nn.Dropout(0.1))
    
    return nn.Sequential(*layers)


class ENC_CNNTurbo_2D(nn.Module):
    def __init__(self, config, interleaver):
        super(ENC_CNNTurbo_2D, self).__init__()
        
        self.config = config
        self.interleaver = interleaver
        
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Encoder CNN and Linear layers for the combined 2D input
        self.enc_cnn = build_encoder_block_2d(config.enc_num_layer, 1, self.config.enc_num_unit, config.enc_kernel_size)  # Setting in_channels to 1
        self.enc_linear = nn.Linear(config.enc_num_unit, 1)

    def set_parallel(self):
        self.enc_cnn = nn.DataParallel(self.enc_cnn)
        self.enc_linear = nn.DataParallel(self.enc_linear)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        
        x_sys = inputs
        x_p1 = self.interleaver.interleave(inputs)
        
        # Stack the sequences along the last dimension and reshape for 2D CNN
        x_combined = torch.stack([x_sys.squeeze(-1), x_p1.squeeze(-1)], dim=-1).unsqueeze(1)
        print("before cnn",x_combined.shape)
        x_combined = self.enc_cnn(x_combined)
        print("after cnn",x_combined.shape)
        
        x_tx = F.elu(self.enc_linear(x_combined))

        codes = self.power_constraint(x_tx)

        return codes.squeeze(dim=2)