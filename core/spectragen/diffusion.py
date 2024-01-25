import torch.nn as nn
import torch


class feedforward(nn.Module):
    def __init__(self, embed_size, output_size=None):
        super(feedforward, self).__init__()
        if output_size is None:
            output_size = embed_size
        self.linear = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.GELU(),
            nn.Linear(embed_size * 2, output_size),
        )

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class attention(nn.Module):
    def __init__(self, embed_size, heads_num):
        super(attention, self).__init__()
        self.embed_size = embed_size
        self.heads_num = heads_num
        self.head_dim = embed_size // heads_num

        if self.head_dim * heads_num != embed_size:
            raise ValueError(
                f"Embedding size needs to be divisible by heads_num {heads_num}"
            )

        self.values_l = nn.Linear(embed_size, embed_size, bias=False)
        self.keys_l = nn.Linear(embed_size, embed_size, bias=False)
        self.queries_l = nn.Linear(embed_size, embed_size, bias=False)
        self.mh_l = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, values_in, keys_in, query_in):
        # input size: (N, seq_len, embed_size)
        N = values_in.shape[0]
        value_len = values_in.shape[1]
        key_len = keys_in.shape[1]
        query_len = query_in.shape[1]

        values = self.values_l(values_in)
        keys = self.keys_l(keys_in)
        queries = self.queries_l(query_in)

        values = values.reshape(N, value_len, self.heads_num, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads_num, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads_num, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads_num * self.head_dim
        )
        out = self.mh_l(out)
        return out


class transformerspatial(nn.Module):
    def __init__(self, embed_size, heads, txtinput_size=256, out_size=None):
        super(transformerspatial, self).__init__()
        self.embed_size = embed_size
        self.feedforward_t = feedforward(txtinput_size, embed_size)
        self.attention = attention(embed_size, heads)
        self.norm0 = nn.LayerNorm(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feedforward = feedforward(embed_size, out_size)

    def forward(self, unet_input, text_input):

        text_input = self.feedforward_t(text_input)

        N = unet_input.shape[0]

        l = unet_input.shape[2]

        unet_input = unet_input.view(N, self.embed_size, -1).permute(0, 2, 1)

        # value_key = torch.cat([unet_input, text_input], dim=1)
        value_key = unet_input + text_input
        value_key = self.norm0(value_key)

        unet_input = self.norm1(unet_input)

        attention = self.attention(value_key, value_key, value_key)
        attention = self.norm2(attention + unet_input)
        out = self.feedforward(attention) + unet_input
        out = out.permute(0, 2, 1).view(N, self.embed_size, l)

        return out


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


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
    def __init__(self, in_channels, out_channels, stridex, knsize=3, emb_dim=256):
        super(Down, self).__init__()

        self.ff = feedforward(emb_dim, in_channels)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels),
        )
        padsz = (knsize - 1) // 2
        self.downdim = nn.Conv1d(out_channels, out_channels, kernel_size=knsize, stride=stridex, padding=padsz, bias=False)

    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor):
        t = self.ff(t_embedding).reshape(x.shape[0], -1, 1)
        out = self.conv(x + t)
        out = self.downdim(out)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels, stridex, knsize=3, emb_dim=256):
        super(Up, self).__init__()
        padsz = (knsize - 1) // 2
        self.updim = nn.ConvTranspose1d(
            in_channels, in_channels,
            kernel_size=knsize, stride=stridex, padding=padsz,
            bias=False, output_padding=stridex - 1)

        self.ff = feedforward(emb_dim, in_channels + in_channels_skip)
        self.conv = nn.Sequential(
            DoubleConv(in_channels + in_channels_skip, out_channels),
            DoubleConv(out_channels, out_channels),
        )

    def forward(self, x, x_skip, t_embedding):
        t = self.ff(t_embedding).reshape(x.shape[0], -1,1)
        x_up = self.updim(x)
        out = torch.cat((x_skip, x_up), dim=1)
        out = self.conv(out + t)

        return out


class UNet(nn.Module):
    def __init__(self, noise_steps, time_dim=256, txtinput_size=256, channel_all=320):
        super(UNet, self).__init__()
        self.time_dim = time_dim

        self.fftxt = feedforward(txtinput_size, txtinput_size)

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(noise_steps, time_dim)
        self.time_embed.weight.data = sinusoidal_embedding(noise_steps, time_dim)
        self.time_embed.requires_grad_(False)

        channel_mid = channel_all // 2
        channel_qt = channel_mid // 2
        channel_d8 = channel_qt // 2
        head_num = 10

        # Left side
        self.input_conv = DoubleConv(1, channel_d8)
        self.convdown0 = Down(channel_d8, channel_qt, 5,5)
        self.trandown0 = transformerspatial(channel_qt, head_num, txtinput_size)
        self.convdown1 = Down(channel_qt, channel_mid, 3,5)
        self.trandown1 = transformerspatial(channel_mid, head_num, txtinput_size)
        self.convdown2 = Down(channel_mid, channel_all, 2)
        self.trandown2 = transformerspatial(channel_all, head_num, txtinput_size)

        # Bottleneck
        self.ffmid = feedforward(time_dim, channel_all)
        self.convmid0 = DoubleConv(channel_all, channel_all, channel_all,3)
        self.convmid1 = DoubleConv(channel_all, channel_all, channel_all,3)

        # Right side
        self.tranup0 = transformerspatial(channel_all, head_num, txtinput_size)
        self.convup0 = Up(channel_all, channel_mid, channel_mid, 2)
        self.tranup1 = transformerspatial(channel_mid, head_num, txtinput_size)
        self.convup1 = Up(channel_mid, channel_qt, channel_qt, 3,5)
        self.tranup2 = transformerspatial(channel_qt, head_num, txtinput_size)
        self.convup2 = Up(channel_qt, channel_d8, channel_d8, 5,5)
        self.out_conv = nn.Sequential(
            nn.Conv1d(channel_d8, channel_d8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(channel_d8, 1, kernel_size=3, stride=1, padding=1)
        )

    def midnet(self, x,t_embedding):
        t_embedding_mid = self.ffmid(t_embedding).reshape(x.shape[0], -1, 1)
        x = self.convmid0(x+t_embedding_mid)
        x = self.convmid1(x+t_embedding_mid)
        return x

    def forward(self, specinput, textinput, t) -> torch.Tensor:

        textinput = self.fftxt(textinput)

        t_embedding = self.time_embed(t)

        x0 = specinput.unsqueeze(1)
        x0 = self.input_conv(x0)

        x0_down = self.convdown0(x0, t_embedding)
        x0_down = self.trandown0(x0_down, textinput)

        x1_down = self.convdown1(x0_down, t_embedding)
        x1_down = self.trandown1(x1_down, textinput)

        x2_down = self.convdown2(x1_down, t_embedding)
        x2_down = self.trandown2(x2_down, textinput)

        x_mid = self.midnet(x2_down, t_embedding)

        x2_up = self.tranup0(x_mid, textinput)
        x2_up = self.convup0(x2_up, x1_down, t_embedding)

        x1_up = self.tranup1(x2_up, textinput)
        x1_up = self.convup1(x1_up, x0_down, t_embedding)

        x0_up = self.tranup2(x1_up, textinput)
        x0_up = self.convup2(x0_up, x0, t_embedding)

        x = self.out_conv(x0_up)
        return x.squeeze(1)

class DDPM(nn.Module):
    def __init__(self, n_steps, min_beta=10 ** -4, max_beta=0.02, device=torch.device('cuda'), image_chw=(1, 2100)):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.Unet = UNet(n_steps).to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        x = x0.unsqueeze(1)
        n, c, l = x.shape
        a_bar = self.alpha_bars[t].to(self.device)

        if eta is None:
            eta = torch.randn(n, c, l).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1) * x + (1 - a_bar).sqrt().reshape(n, 1, 1) * eta
        return noisy

    def backward(self, latent_in, text_in, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.Unet(latent_in, text_in, t)

