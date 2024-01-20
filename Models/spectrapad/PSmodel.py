
import torch.nn as nn
import torch
# import torch.nn.functional as F
import math


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


# attention mechanism
class attention(nn.Module):
    def __init__(self, embed_size, heads_num, token_query=False):
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
        if not token_query:
            self.queries_l = nn.Parameter(torch.randn(embed_size, embed_size))
        self.mh_l = nn.Linear(embed_size, embed_size, bias=False)
        self.token_query = token_query

    def forward(self, values_in, keys_in, query_in, mask_k=None, mask_q=None):
        # input size: (N, seq_len, embed_size)
        N = values_in.shape[0]
        value_len = values_in.shape[1]
        key_len = keys_in.shape[1]
        query_len = query_in.shape[1]

        values = self.values_l(values_in)
        keys = self.keys_l(keys_in)
        if not self.token_query:
            queries = torch.matmul(query_in, self.queries_l)
        else:
            queries = query_in

        values = values.reshape(N, value_len, self.heads_num, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads_num, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads_num, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask_k is None:
            mask_k = torch.ones(N, key_len).bool().to(keys.device)
        if mask_q is None:
            mask_q = torch.ones(N, query_len).bool().to(queries.device)

        mask_k = mask_k.unsqueeze(1)
        mask_q = mask_q.unsqueeze(2)

        mask_energy = mask_k.unsqueeze(1) & mask_q.unsqueeze(1)
        energy = energy.masked_fill(mask_energy == False, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads_num * self.head_dim
        )

        out = out.masked_fill(mask_q == False, 0)
        out = self.mh_l(out)
        return out


class transformerlayer(nn.Module):
    def __init__(self, embed_size, heads, output_size=None, token_query=False):
        super(transformerlayer, self).__init__()
        # self.embed_size = embed_size
        if output_size is None:
            output_size = embed_size
        self.attention = attention(embed_size, heads, token_query)
        self.norm0 = nn.LayerNorm(embed_size)
        self.feedforward = feedforward(embed_size, output_size)
        self.norm1 = nn.LayerNorm(output_size)

    def forward(self, value, key, query, mask_k=None, mask_q=None):

        value_n = self.norm0(value)
        key_n = self.norm0(key)

        attention = self.attention(value_n, key_n, query, mask_k, mask_q)

        if not mask_q is None:
            attention = attention.masked_fill(mask_q.unsqueeze(2) == False, 0)

        out = self.norm1(attention + query)
        out = self.feedforward(out)

        if mask_q is not None:
            out = out * mask_q.unsqueeze(2)

        return out


class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, bandnum=42):
        super(PositionalEncoder, self).__init__()
        self.embed_size = embed_size

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(bandnum, embed_size)
        for pos in range(bandnum):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, spectram, mask):
        # make embeddings relatively larger
        input_embedding = spectram * math.sqrt(self.embed_size)
        # add constant to embedding
        input_embedding = input_embedding + self.pe
        input_embedding = input_embedding * mask.unsqueeze(2)
        return input_embedding


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernelsize = 5, residual=True):
        super(DoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        if not in_channels == out_channels:
            # if residual == True:
            #     print('Warning: residual is not possible')
            residual = False

        self.residual = residual
        padding = (kernelsize - 1) // 2

        self.ln0 = nn.GroupNorm(1, in_channels)
        self.conv0 = nn.Conv1d(in_channels, mid_channels, kernel_size=kernelsize, padding=padding)
        self.ln1 = nn.GroupNorm(1, mid_channels)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv1d(mid_channels, out_channels, kernel_size=kernelsize, padding=padding)

    def forward(self, x):

        out = self.ln0(x)
        out = self.conv0(out)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        if self.residual:
            return x + out

        return out


class SSAE(nn.Module):
    def __init__(self, embedsize, channelsize, headnum, bandwidth=50, bandnum=42):
        super(SSAE, self).__init__()

        h_channelsize = channelsize // 2
        q_channelsize = channelsize // 4

        self.bandwidth = bandwidth
        self.bandnum = bandnum

        # Embedding layers
        self.positionencoding = PositionalEncoder(bandwidth, bandnum)
        self.project = nn.Linear(bandwidth, embedsize, bias=False)
        self.token_emb = nn.Parameter(torch.rand(1, bandnum, embedsize))
        self.transformer_emb0 = transformerlayer(embedsize, headnum, embedsize)
        self.transformer_emb1 = transformerlayer(embedsize, headnum, embedsize)
        self.transformer_emb2 = transformerlayer(embedsize, headnum, embedsize, True)

        # Encoder layers

        self.conv_down0 = nn.Sequential(
            DoubleConv(embedsize // bandwidth, q_channelsize),
            nn.GELU(),
            nn.Conv1d(q_channelsize, h_channelsize, kernel_size=53, stride=2, padding=0),
        )

        self.conv_en0 = nn.Sequential(
            DoubleConv(h_channelsize, h_channelsize),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.conv_en1 = nn.Sequential(
            DoubleConv(h_channelsize, h_channelsize),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.down1 = nn.Sequential(
            DoubleConv(h_channelsize, channelsize),
            nn.GELU(),
            nn.Conv1d(channelsize, channelsize, kernel_size=3, stride=2, padding=1),
        )

        self.conv_en2 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            DoubleConv(channelsize, channelsize),
        )

        self.conv_en3 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            DoubleConv(channelsize, channelsize),
        )

        self.down2 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            nn.GELU(),
            nn.Conv1d(channelsize, channelsize, kernel_size=3, stride=2, padding=1),
        )

        self.conv_en4 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            DoubleConv(channelsize, channelsize),
        )

        self.conv_en5 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            DoubleConv(channelsize, 1,channelsize,3),
        )

        # Decoder layers

        self.conv_de0 = nn.Sequential(
            DoubleConv(1, channelsize,channelsize,3),
            DoubleConv(channelsize, channelsize),
        )

        self.conv_de1 = nn.Sequential(
            DoubleConv(channelsize, channelsize),
            DoubleConv(channelsize, channelsize),
        )

        self.conv_up0 = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(channelsize, h_channelsize, kernel_size=16, stride=2),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.conv_de2 = nn.Sequential(
            DoubleConv(h_channelsize, h_channelsize),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.conv_de3 = nn.Sequential(
            DoubleConv(h_channelsize, h_channelsize),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.conv_de4 = nn.Sequential(
            DoubleConv(h_channelsize, h_channelsize),
            DoubleConv(h_channelsize, h_channelsize),
        )

        self.conv_de5 = nn.Sequential(
            DoubleConv(h_channelsize, q_channelsize),
            DoubleConv(q_channelsize, 1, q_channelsize, 3),
        )

    def embedding(self, spectra):
        # convert spectra vector to spectra matrix
        spectram = spectra.reshape(spectra.shape[0], self.bandnum, self.bandwidth)  # shape: (batch, bandnum, bandwidth)

        # get mask
        mask = (spectram.sum(dim=2) != 0)

        # positional encoding
        spectra_embedding = self.positionencoding(spectram, mask)

        # linear transformation
        spectra_embedding = self.project(spectra_embedding)  # shape: (batch, bandnum, embedsize)

        # transformer layer 0
        out_trans0 = self.transformer_emb0(spectra_embedding, spectra_embedding, spectra_embedding, mask, mask)

        # transformer layer 1
        out_trans1 = self.transformer_emb1(out_trans0, out_trans0, out_trans0, mask, mask) + out_trans0

        querytoken = self.token_emb.repeat(spectra_embedding.shape[0], 1, 1)
        out_trans1 = self.transformer_emb2(out_trans1, out_trans1, querytoken, mask)

        # reshape
        out_specembedding = out_trans1.reshape(out_trans1.shape[0], self.bandwidth * self.bandnum, -1)
        out_specembedding = out_specembedding.permute(0, 2, 1)  # shape: (batch, embedsize//bandwidth, wavelenths)

        return out_specembedding

    def encoder(self, spectra_embedding):
        # downsample layer 0
        out = self.conv_down0(spectra_embedding)

        # convolution layer 0
        out = self.conv_en0(out) + out

        # convolution layer 1
        out = self.conv_en1(out) + out

        # downsample layer 1
        out = self.down1(out)

        # convolution layer 2
        out = self.conv_en2(out) + out

        # convolution layer 3
        out = self.conv_en3(out) + out

        # downsample layer 2
        out = self.down2(out)

        # convolution layer 4
        out = self.conv_en4(out) + out

        # convolution layer 5
        out = self.conv_en5(out)

        # normalization
        out = (out - out.mean(dim=2, keepdim=True)) / out.std(dim=2, keepdim=True)

        return out

    def decoder(self, latent):

        # convolution layer 0
        out = self.conv_de0(latent)

        # convolution layer 1
        out = self.conv_de1(out) + out

        # upsample layer 0
        out = self.conv_up0(out)

        # convolution layer 2
        out = self.conv_de2(out) + out

        # convolution layer 3
        out = self.conv_de3(out) + out

        # convolution layer 4
        out = self.conv_de4(out) + out

        # convolution layer 5
        outspectra = self.conv_de5(out)

        outspectra = outspectra.squeeze(1)

        return outspectra

    def forward(self, spectra):

        spectra_embedding = self.embedding(spectra)

        latent = self.encoder(spectra_embedding)

        out = self.decoder(latent)

        return out
