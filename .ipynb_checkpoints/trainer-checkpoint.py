import math
import logging

from tqdm.notebook import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

import matplotlib.pyplot as plt

import ntd as ntd

#####
import random



class TrainerConfig:
    # Default optimization parameters, gets overriden by config init
    max_epochs = 10
    learning_rate = 1e-4
    iterations=100
    rate = 0.5
    optimizer = "Adam"

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

OPTIMIZERS = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'NAdam': optim.NAdam,
}
               

class Trainer:
    def __init__(self, encoder, decoder, channel, config):
        self.channel, self.encoder, self.decoder, self.config= channel, encoder, decoder, config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_validation_loss(self):
        # Only to check BER, SER without special encoder/ decoder handling
        with torch.no_grad():
            ber, ser, loss_val = [], [], []
            self.encoder.eval()
            self.decoder.eval()
            for _ in range(100):
                m_val = torch.randint(0, 2, (self.config.batchsize, self.config.coderate), dtype=torch.float).to(self.device) 
                x_val = self.encoder(m_val)
                x_out_val = self.channel(x_val, self.config.noise_std)
                output_val = self.decoder(x_out_val)

                loss_val.append(F.binary_cross_entropy(output_val, m_val).item())
                ber.append(utils.error_binary(torch.round(output_val), m_val)[0])
                ser.append(utils.error_binary(torch.round(output_val), m_val)[1])
            print("------------------")
            print(f"(Validation) : loss {np.mean(loss_val):.3e} BER {np.mean(ber):.3e} SER {np.mean(ser):.3e}")
            print("------------------")
            self.encoder.train()
            self.decoder.train()
        return np.mean(loss_val)
        
    def train_alternate(self):
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        lr = config.learning_rate
        optimizer = OPTIMIZERS[config.optimizer]
        
        #plot_grad = False
        #
        #if plot_grad: 
        #    grad_norms = {name: [] for name, _ in encoder.named_parameters() if _.requires_grad}
        #    grad_norms2 = {name: [] for name, _ in decoder.named_parameters() if _.requires_grad}
        #    weight_norms = {name: [] for name, _ in encoder.named_parameters() if _.requires_grad}
        #    weight_norms2 = {name: [] for name, _ in decoder.named_parameters() if _.requires_grad}
        
        encoder_optimizer = optimizer(encoder.parameters(), lr=lr)
        decoder_optimizer = optimizer(decoder.parameters(), lr=lr)
        #scheduler_enc = StepLR(encoder_optimizer, step_size=30, gamma=0.5) # Every 30 Epochs, reduce lr by factor 0.75
        #scheduler_dec = StepLR(decoder_optimizer, step_size=30, gamma=0.5)
        
        loss_CE = nn.CrossEntropyLoss()
        loss_epoch = []
        best_loss = float('inf')
        
        def run_epoch():
            loss_tot, losses, batch_BER, batch_SER, bler_list = [], [], [], [], []
            
            def train_(encoder_optimizer=None, decoder_optimizer=None):
                
                m = torch.randint(0, 2, (config.batchsize, config.coderate), dtype=torch.float).to(self.device)
                x = encoder(m)

                if decoder_optimizer:  # Adding noise matrix only for decoder training
                    #x = x.detach()
                    decoder_optimizer.zero_grad()
                    low_snrdb = config.training_snr - 3.5
                    high_snrdb = config.training_snr + 0
                    snrdb_matrix = np.random.uniform(low_snrdb, high_snrdb, size=x.shape)
                    noise_std_matrix = utils.EbNo_to_noise(snrdb_matrix, config.rate)
                    noise_std_matrix_torch = torch.from_numpy(noise_std_matrix).float().to(self.device)
                    x_out = channel(x, noise_std_matrix_torch)
                else:
                    encoder_optimizer.zero_grad()
                    x_out = channel(x, config.noise_std)

                output = decoder(x_out)

                loss = F.binary_cross_entropy(output, m)
                #loss = loss.mean()
                loss.backward()
                
                #if plot_grad:
                #    # Store gradient norms
                #    for name, param in encoder.named_parameters():
                #        if param.requires_grad:
                #            grad_norm = param.grad.data.norm(2).item()
                #            grad_norms[name].append(grad_norm)
                #            weight_norm = param.data.norm(2).item()
                #            weight_norms[name].append(weight_norm)
                #    for name, param in decoder.named_parameters():
                #        if param.requires_grad:
                #            grad_norm = param.grad.data.norm(2).item()
                #            grad_norms2[name].append(grad_norm)
                #            weight_norm = param.data.norm(2).item()
                #            weight_norms2[name].append(weight_norm)
                            
                #----
                # For gradient accumulation for large batch sizes
                #----
                #if (i+ 1) % 10 == 0:
                    #optimizer.step()  # Perform a weight update
                    #optimizer.zero_grad()  # Zero out the accumulated gradients
                    
                if encoder_optimizer:
                    encoder_optimizer.step()
                    #encoder_optimizer.zero_grad()
                if decoder_optimizer:
                    decoder_optimizer.step()
                    #decoder_optimizer.zero_grad()

                loss_tot.append(loss.item())
                batch_BER.append(utils.error_binary(torch.round(output), m)[0])
                batch_SER.append(utils.error_binary(torch.round(output), m)[1])
            
            # Train encoder for all iterations
            enc_its = 1
            for _ in range(enc_its):
                for i in (pbar := tqdm(range(config.iterations))):
                    train_(encoder_optimizer=encoder_optimizer)
                    pbar.set_description(f"epoch {epoch+1} (Encoder {_+1}/{enc_its}) : loss {np.mean(loss_tot[-100:]):.3e} BER {np.mean(batch_BER[-100:]):.3e} SER {np.mean(batch_SER):.3e}")

            # Reset metrics
            loss_tot, batch_BER, batch_SER, bler_list = [], [], [], []

            # Train decoder for 5 times all iterations
            dec_its = 5
            for _ in range(dec_its):
                for i in (pbar := tqdm(range(config.iterations))):
                    train_(decoder_optimizer=decoder_optimizer)
                    pbar.set_description(f"epoch {epoch+1} (Decoder {_+1}/{dec_its}) : loss {np.mean(loss_tot[-100:]):.3e} BER {np.mean(batch_BER[-100:]):.3e} SER {np.mean(batch_SER):.3e}")
                    
            loss_val = self.compute_validation_loss()

            return loss_val#, (grad_norms, grad_norms2, weight_norms, weight_norms2)

    
        for epoch in range(config.max_epochs):
            
            loss_tot = run_epoch()
            loss_tot_avg = np.mean(loss_tot)
            loss_epoch.append(loss_tot_avg)
            #scheduler_enc.step()
            #scheduler_dec.step()
            
            if loss_tot_avg < best_loss:
                best_loss = loss_tot_avg
                torch.save({ 'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, config.path)
                
        return loss_epoch#, data
    
    def train(self): # joint training
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        lr = config.learning_rate
        combined_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = OPTIMIZERS[config.optimizer]
        optimizer = optimizer(combined_params, lr=lr)

        loss_epoch = []
        best_loss = float('inf')
    
        def run_epoch():         
            loss_tot, losses, batch_BER, batch_SER, bler_list = [], [], [], [], []

            for i in (pbar := tqdm(range(config.iterations))):  # Training loop
        
                m = torch.randint(0, 2, (config.batchsize, config.coderate), dtype=torch.float)
                m = m.to(self.device)
                x = encoder(m)
                x_out = channel(x, config.noise_std)
                output = decoder(x_out)
            
                loss = F.binary_cross_entropy(output, m)                
                loss.backward()
                
                #----
                # For gradient accumulation
                #----
                #if (i+ 1) % 10 == 0:
                #    optimizer.step()  # Perform a weight update
                #    optimizer.zero_grad()  # Zero out the accumulated gradients
                
                optimizer.step()
                optimizer.zero_grad()
                
                loss_tot.append(loss.item())
                  
                batch_BER.append(utils.error_binary(torch.round(output), m)[0])
                batch_SER.append(utils.error_binary(torch.round(output), m)[1])
                                
                pbar.set_description(f"epoch {epoch+1} : loss {np.mean(loss_tot[-100:]):.3e} BER {np.mean(batch_BER[-100:]):.3e} SER {np.mean(batch_SER):.3e}")
            loss_val = self.compute_validation_loss()

            return loss_tot
            
        
        for epoch in range(config.max_epochs):
            loss_tot = run_epoch() 
            loss_tot_avg = np.mean(loss_tot)
            loss_epoch.append(loss_tot_avg)
            
            if loss_tot_avg < best_loss:
                best_loss = loss_tot_avg
                torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, config.path)
                
        return loss_epoch

    
    def test(self, snr_range, rate, iterations):
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        ber, bler, ser = [], [], []
        its = int(iterations)
        
        encoder.eval()
        decoder.eval()
        
        def run_epoch():         
            batch_BER, batch_SER, bler_list = [], [], []
            noise_std = utils.EbNo_to_noise(snr_range[epoch], rate)            
            
            for it in (pbar := tqdm(range(its))):
                
                m = torch.randint(0, 2, (config.batchsize, config.coderate), dtype=torch.float).to(self.device)
                
                with torch.no_grad():  # Disable gradient calculation
                    x = encoder(m)
                    x_out = channel(x, noise_std)
                    output = decoder(x_out)

                batch_BER.append(utils.error_binary(torch.round(output), m)[0])
                batch_SER.append(utils.error_binary(torch.round(output), m)[1])
                pbar.set_description(f"SNR: {snr_range[epoch]} BER {np.mean(batch_BER):.3e} SER {np.mean(batch_SER):.3e}")
                    
       
            return np.mean(batch_BER), np.mean(bler_list), np.mean(batch_SER)
            
        for epoch in range(len(snr_range)):
            
            ber_i, bler_i, ser_i = run_epoch()
            ber.append(ber_i)
            bler.append(bler_i)
            ser.append(ser_i) 
        
        encoder.train()
        decoder.train()
        
        return ber, ser, bler
    
# Experimental NTD optimizer
    def train_NTD(self):
        channel, config, encoder, decoder = self.channel, self.config, self.encoder, self.decoder
        lr = config.learning_rate
        combined_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = ntd.NTD(combined_params, opt_f=np.inf, adaptive_grid_size=False, use_trust_region=True, s_scale_factor=1e-6, verbose=False)#torch.optim.NAdam(combined_params, lr=lr)

        loss_epoch = []
        best_loss = float('inf')
    
        def run_epoch():         
            loss_tot, losses, batch_BER, batch_SER, bler_list = [], [], [], [], []

            for i in (pbar := tqdm(range(100))):  # Training loop
                current_loss = None
                def closure():
                    optimizer.zero_grad()               
                    m = torch.randint(0, 2, (config.batchsize, config.coderate), dtype=torch.float).to(self.device)   
                    x = encoder(m)
                    x_out = channel(x, config.noise_std)
                    output = decoder(x_out)
                    loss = F.binary_cross_entropy(output, m)
                    loss.backward()
                    current_loss = loss.item()
                    return loss
                

                optimizer.step(closure)
                
                
            loss_val = self.compute_validation_loss()

            return loss_tot
            
        
        for epoch in range(config.max_epochs):
            loss_tot = run_epoch() # pushing batch size higher with epochs
            loss_tot_avg = np.mean(loss_tot)
            loss_epoch.append(loss_tot_avg)
            
            if loss_tot_avg < best_loss:
                best_loss = loss_tot_avg
                torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, config.path)
                
        return loss_epoch