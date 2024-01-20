import Models.spectrapad.PSmodel as PS
import torch
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def padspectra(spectra, device= torch.device("cuda"),padgap = 50,wavelength=torch.arange(400, 2500, 1)):
    device_o = spectra.device
    embedsize = 200
    channelsize = 120
    headnum = 10
    # Set default wavelength
    wavelength4 = torch.cat((wavelength[::4], wavelength[-1:]),dim=0)  # Note the change in dim from 1 to 0 for 1D tensor

    bandwidth = 50
    bandnum = len(wavelength) // bandwidth  # 42

    # Instantiate the model
    PSmodel = PS.SSAE(embedsize, channelsize, headnum, bandwidth, bandnum).to(device)
    PSmodel.load_state_dict(torch.load('/home/tlei/PycharmProjects/SOLGM/Models/spectrapad/PS_para_22934.pth', map_location=device))
    PSmodel.eval()

    spectra_ts = spectra.clone().detach().to(device)

    spectra_pad = PSmodel(spectra_ts)
    f = interp1d(wavelength4, spectra_pad.detach().cpu().numpy(), kind='cubic')
    spectra_pad_cpu = f(wavelength)

    mask1 = (spectra > 0) & (spectra < 1)
    spectra_pad_cpu[mask1] = spectra[mask1]
    spectra_pad_np = savgol_filter(spectra_pad_cpu, window_length=100, polyorder=2)

    spectra_pad_sn = torch.tensor([]).to(device)
    for irow in range(spectra_ts.shape[0]):
        spectrum_ts = spectra_ts[irow, :].float().to(device)
        spectrum_pad_np = torch.tensor(spectra_pad_np[irow, :]).float().to(device)
        maskidx = spectrum_ts.nonzero(as_tuple=True)[0]
        maskidx, _ = torch.sort(maskidx)
        if maskidx[0] > 0:
            startidx = maskidx[:padgap]
            startratio = torch.linspace(1, 0, padgap).to(device)
            spectrum_pad_np[startidx] = spectrum_pad_np[startidx] * startratio + spectrum_ts[startidx] * (1 - startratio)
            maskidx = maskidx[padgap:]
        if maskidx[-1] < 2099:
            endidx = maskidx[-padgap:]
            endratio = torch.linspace(0, 1, padgap).to(device)
            spectrum_pad_np[endidx] = spectrum_pad_np[endidx] * endratio + spectrum_ts[endidx] * (1 - endratio)
            maskidx = maskidx[:-padgap]

        spectrum_pad_np[maskidx] = spectrum_ts[maskidx]
        # concatenate
        spectra_pad_sn = torch.cat((spectra_pad_sn, spectrum_pad_np.unsqueeze(0).to(device)), dim=0)

    spectra_n = spectra_pad_sn.to(device_o)

    return spectra_n



