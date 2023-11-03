import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from numpy import arange
from numpy.random import mtrand


class Channel_AWGN(torch.nn.Module):
    def __init__(self):
        super(Channel_AWGN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, codeword, noise_std):
         
        channel_noise = noise_std*torch.randn(codeword.shape).to(self.device) # X = sigma*Z, where Z is the standard normal N(0,1) and sigma the standard deviation, and X = N(0, sigma)
        
        rec_signal = codeword + channel_noise

        return rec_signal

    
class Channel_burst(torch.nn.Module):
    def __init__(self):
        super(Channel_burst, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, codeword, noise_std):
        
        i,j = codeword.size()
        
        channel_noise = noise_std*torch.randn(codeword.shape).to(self.device)
        burst = np.sqrt(2)*channel_noise
        burst_prob = torch.bernoulli(torch.tensor(.01)).to(self.device)
        
        rec_signal = codeword + channel_noise + burst*burst_prob

        return rec_signal