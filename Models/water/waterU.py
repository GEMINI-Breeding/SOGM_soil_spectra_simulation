import torch.nn as nn
import torch
from scipy.signal import savgol_filter

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=5,):
        super(DoubleConv, self).__init__()
        padding = (kernel_size - 1) // 2
        if not mid_channels:
            mid_channels = out_channels
        self.ln = nn.GroupNorm(1, mid_channels)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x, residual=False):
        out = self.conv1(x)
        out = self.ln(out)
        out = self.gelu(out)
        out = self.conv2(out)

        if residual:
            return self.gelu(x + out)

        return self.gelu(out)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, stridex, knsize=3):
        super(Down, self).__init__()

        self.emb_t = nn.Linear(1, in_channels)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels),
        )
        padsz = (knsize - 1) // 2
        self.downdim = nn.Conv1d(out_channels, out_channels, kernel_size=knsize, stride=stridex, padding=padsz, bias=False)

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor):
        t = self.emb_t(t_embedding).reshape(x.shape[0], -1, 1)
        out = self.conv(x + t)
        out = self.downdim(out)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, stridex, knsize=3):
        super(Up, self).__init__()
        padsz = (knsize - 1) // 2
        self.updim = nn.ConvTranspose1d(
            in_channels, in_channels,
            kernel_size=knsize, stride=stridex, padding=padsz,
            bias=False, output_padding=stridex - 1)

        self.emb_t = nn.Linear(1, in_channels + in_channels_skip)
        self.conv = nn.Sequential(
            DoubleConv(in_channels + in_channels_skip, out_channels),
            DoubleConv(out_channels, out_channels),
        )

    def forward(self, x, x_skip, t_embedding):
        t = self.emb_t(t_embedding).reshape(x.shape[0], -1,1)
        x_up = self.updim(x)
        out = torch.cat((x_skip, x_up), dim=1)
        out = self.conv(out + t)

        return out


class WaterUNet(nn.Module):
    def __init__(self, channel_all=64):
        super(WaterUNet, self).__init__()


        channel_mid = channel_all // 2
        channel_qt = channel_mid // 2
        channel_d8 = channel_qt // 2

        # Left side
        self.input_conv = DoubleConv(1, channel_d8)
        self.convdown0 = Down(channel_d8, channel_qt, 5,5)
        self.convdown1 = Down(channel_qt, channel_mid, 3,5)
        self.convdown2 = Down(channel_mid, channel_all, 2)


        # Bottleneck
        self.emb_t_m = nn.Linear(1, channel_all)
        self.convmid0 = DoubleConv(channel_all, channel_all, channel_all,3)
        self.convmid1 = DoubleConv(channel_all, channel_all, channel_all,3)

        # Right side
        self.convup0 = Up(channel_all, channel_mid, channel_mid, 2)
        self.convup1 = Up(channel_mid, channel_qt, channel_qt, 3,5)
        self.convup2 = Up(channel_qt, channel_d8, channel_d8, 5,5)
        self.out_conv = nn.Sequential(
            nn.Conv1d(channel_d8, channel_d8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(channel_d8, 1, kernel_size=3, stride=1, padding=1)
        )

    def midnet(self, x,t_embedding):
        t_embedding_mid = self.emb_t_m(t_embedding).reshape(x.shape[0], -1, 1)
        x = self.convmid0(x+t_embedding_mid)
        x = self.convmid1(x+t_embedding_mid)
        return x

    def forward(self, specinput, t_embedding) -> torch.Tensor:


        x0 = specinput.unsqueeze(1)
        x0 = self.input_conv(x0)

        x0_down = self.convdown0(x0, t_embedding)

        x1_down = self.convdown1(x0_down, t_embedding)

        x2_down = self.convdown2(x1_down, t_embedding)

        x_mid = self.midnet(x2_down, t_embedding)

        x2_up = self.convup0(x_mid, x1_down, t_embedding)

        x1_up = self.convup1(x2_up, x0_down, t_embedding)

        x0_up = self.convup2(x1_up, x0, t_embedding)

        x = self.out_conv(x0_up)
        return x.squeeze(1)

def modelwater(dryspectra,SMCs,device='cpu'):
    device_ = torch.device(device)
    WUtest = WaterUNet(80).to(device_)
    WUtest.load_state_dict(torch.load('/home/tlei/PycharmProjects/SOLGM/Models/water/WU_para_5200.pth', map_location=torch.device(device_)))
    WUtest.eval()

    wet_spectra_dff = WUtest(dryspectra.clone().to(device_), SMCs.clone().to(device_))
    # make diff non-negative
    wet_spectra_dff = wet_spectra_dff.clamp(min=0)
    wet_spectra = dryspectra.clone().to(device_) - wet_spectra_dff
    # make wet_spectra non-negative
    wet_spectra = wet_spectra.clamp(min=0)
    wet_spectra_np = wet_spectra.cpu().detach().numpy()
    wet_spectra_smooth_m = savgol_filter(wet_spectra_np, window_length=50, polyorder=2, axis=1)
    wet_spectra_smooth_se = savgol_filter(wet_spectra_np, window_length=150, polyorder=2, axis=1)
    wet_spectra_smooth_se[:,200:1700] = wet_spectra_smooth_m[:,200:1700]
    wet_spectra = torch.from_numpy(wet_spectra_smooth_se).to(device_)
    wet_spectra = wet_spectra.clamp(min=0)
    wet_spectra[wet_spectra>dryspectra] = dryspectra[wet_spectra>dryspectra]
    return wet_spectra