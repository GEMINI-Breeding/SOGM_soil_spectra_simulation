import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Models.spectragen import SPEncoder as PEn
from Models.spectragen import SOD_model as SOD
import os

def initializemodel(device):
    number_index = 0 # define the index of the number token
    num_uniqwords = 509 # define the number of unique words in the vocabulary
    pad_index = 509  # define the padding index (last index of the vocabulary)
    vocab_size = 510  # define the vocabulary size (add 1 for padding)
    embeddingsz_t = 250
    head_num = 10

    Pencoder = PEn.SPEncod(embeddingsz_t, head_num, vocab_size, pad_index, number_index, 256, 16).to(device)
    current_directory = os.getcwd()
    weights_pe_folder = os.path.join(current_directory, 'Models/spectragen/PE_para.pth')
    Pencoder.load_state_dict(torch.load(weights_pe_folder, map_location=torch.device(device)))
    Pencoder.to(device)
    Pencoder.eval()

    n_steps = 300 # define the number of steps for the diffusion process
    diffusionmodel = SOD.DDPM(n_steps, min_beta=10 ** -4, max_beta=0.02,device=device).to(device)
    current_directory = os.getcwd()
    weights_df_folder = os.path.join(current_directory, 'Models/spectragen/DF_para.pth')
    diffusionmodel.load_state_dict(torch.load(weights_df_folder, map_location=torch.device(device)))
    diffusionmodel.to(device)
    diffusionmodel.eval()

    return Pencoder, diffusionmodel

def writespectra(wavelength,spectra_gen, filename):
    x_limit = (400, 2500)
    y_limit = (-150, 250)
    x_label = 'Wavelength (nm)'
    y_label = 'Reflectance (%)'
    title = ''

    plt.figure(figsize=(8, 7))  # Optional: Set the figure size
    plt.plot(wavelength, 100*spectra_gen[:, :].T)

    fontsz = 20
    plt.xlabel(x_label, fontsize=fontsz)
    plt.ylabel(y_label, fontsize=fontsz)
    # plt.title(title, fontsize=fontsz)
    # plt.legend([''], fontsize=fontsz)
    # Set axis limits
    plt.xlim(x_limit)
    plt.ylim(y_limit)

    # Set ticks fontsize
    plt.xticks(fontsize=fontsz)
    plt.yticks(fontsize=fontsz)
    plt.yticks([-100, 0, 100, 200])
    # plt.yticks([])
    plt.xticks([500, 1000, 1500, 2000])
    # plt.xticks([])
    plt.tight_layout()
    # Save the plot as an image file
    plt.savefig(filename, format='jpeg', dpi=300)  # dpi is optional, for resolution adjustment
    plt.close()  # Close the figure

def reversediffusion(diffusionmodel, textemb_input, n_samples, l, series = False,plotprocess = None):

    with torch.no_grad():

        device = diffusionmodel.device

        # Initialize random noise
        if series:
            x = torch.randn(1, l)
            x = x.repeat(n_samples, 1)
            x = x.to(device)
        else:
            x = torch.randn(n_samples, l).to(device)

        for idx, t in enumerate(list(range(diffusionmodel.n_steps))[::-1]):

            # Estimating noise to be removed

            t_step = (torch.ones(n_samples) * t).to(device).long()
            eta_theta = diffusionmodel.backward(x.clone(), textemb_input.clone(), t_step)

            alpha_t = diffusionmodel.alphas[t]
            alpha_t_bar = diffusionmodel.alpha_bars[t]

            # Partially denoising the spectra
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            if plotprocess is not None:
                if t in plotprocess:
                    specfilename ='results/spec_'+str(t)+'.jpeg'
                    writespectra(torch.arange(400, 2500, 2), x.cpu().numpy(), specfilename)

            if t > 0:
                if series:
                    z = torch.randn(1, l)
                    z = z.repeat(n_samples, 1)
                    z = z.to(device)
                else:
                    z = torch.randn(n_samples, l).to(device)


                beta_t = diffusionmodel.betas[t]
                prev_alpha_t_bar = diffusionmodel.alpha_bars[t - 1] if t > 0 else diffusionmodel.alphas[0]
                beta_tilda_t = ((1 - prev_alpha_t_bar) / (1 - alpha_t_bar)) * beta_t
                sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z



        return x

def generatespectra(textidx_input, device,series = False, wavelength =  torch.arange(400, 2500, 1)):

    Pencoder, diffusionmodel = initializemodel(device)
    n_samples = textidx_input.shape[0]
    textidx_input_t = textidx_input.clone().to(device)
    textemb_input = Pencoder(textidx_input_t)
    l = len(wavelength)
    with torch.no_grad():
        l_half = l // 2
        spectra_gen = reversediffusion(diffusionmodel, textemb_input.clone(), n_samples, l_half, series)
    spectra_gen = spectra_gen.cpu().numpy()
    spectra_gen_last = spectra_gen[:, -1]
    wavelength_o = wavelength[::2]
    f = interp1d(wavelength_o, spectra_gen, kind='cubic', bounds_error=False, fill_value="extrapolate")
    spectra_gen = f(wavelength)
    spectra_gen[:, -1] = spectra_gen_last

    return spectra_gen

def plotgeneration(textidx_input, device,series = False, plotprocess = None, wavelength =  torch.arange(400, 2500, 1)):

    Pencoder, diffusionmodel = initializemodel(device)
    n_samples = textidx_input.shape[0]
    textidx_input_t = textidx_input.clone().to(device)
    textemb_input = Pencoder(textidx_input_t)
    l = len(wavelength)
    with torch.no_grad():
        l_half = l // 2
        reversediffusion(diffusionmodel, textemb_input.clone(), n_samples, l_half, series, plotprocess)
