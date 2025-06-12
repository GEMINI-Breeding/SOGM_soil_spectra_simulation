import core.spectrapad.padmodel as PS
import torch
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os

def run_padmodel(spectra, device= torch.device("cuda")):
    # add 50 zeros to the start of the spectra
    spectra = torch.cat((torch.zeros(spectra.shape[0], 50), spectra), dim=1)
    bandnum = 43
    bandwidth = 50
    vocab_size = 27
    hidden_dim = 160
    pad_index = 26
    maxwords = 4
    spectra_input = spectra.reshape(spectra.shape[0],bandnum, bandwidth)  # shape: (batch, bandnum, bandwidth)
    propdata_input = torch.tensor([14.,  0., 17., 26.]).unsqueeze(0).unsqueeze(0).repeat(spectra_input.shape[0],1,1).to(device)

    areg = PS.AutoRegression(8,8,vocab_size, pad_index, bandwidth=bandwidth, maxwords = maxwords, bandnum=bandnum, moe_noise_std = 0.,hidden_dim=hidden_dim)

    # load the model state
    areg.load_state_dict(torch.load('core/spectrapad/spectra_pad_param.pth', map_location=device))
    areg.eval()

    spectra_padded = areg.pad(spectra_input, propdata_input,smoothing=True)
    spectra_padded = spectra_padded.reshape(spectra_padded.shape[0],-1)
    spectra_padded = spectra_padded[:,50:]
    return spectra_padded



