
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class TopGate(nn.Module):
    def __init__(self, input_dim, num_experts, noise_std=1.0, num_gate=1):
        super().__init__()
        self.w_gate = nn.Linear(input_dim, num_experts)
        self.noise_std = noise_std
        if num_gate>num_experts:
            raise ValueError('Number of gate should not greater than number of experts')
        elif num_gate==0:
            raise ValueError('Number of gate should not be 0.')
        self.num_gate = num_gate
    def forward(self, x):
        logits = self.w_gate(x)  # (B, T, E)
        if self.noise_std > 0:
            logits += torch.randn_like(logits) * self.noise_std

        # Get top-K expert indices and logits (K = num_gate)
        topk_scores, topk_idx = torch.topk(logits, k=self.num_gate, dim=-1)
        topk_weights = torch.ones_like(topk_scores) if self.num_gate == 1 else torch.softmax(topk_scores, dim=-1)
        return topk_idx, topk_weights

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, noise_std=1.0,topgate=2):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        # self.gate = Top1Gate(input_dim, num_experts, noise_std)
        self.gate = TopGate(input_dim, num_experts, noise_std,topgate)
        self.output_dim = output_dim

    def forward(self, x):
        B, T, D = x.shape # (batch_size, sequence_length, input_dim)
        x_flat = x.view(-1, D)  # (B*T, D)

        topk_idx, topk_weights = self.gate(x)
        K = self.gate.num_gate
        topk_idx = topk_idx.view(-1, K)
        topk_weights = topk_weights.view(-1, K)

        expert_outputs = torch.zeros(x_flat.size(0), self.output_dim, device=x.device)

        for expert_id in range(self.num_experts):
            # (B*T, K) -> (B*T,) boolean mask where expert_id appears
            mask_expert = (topk_idx == expert_id)  # (B*T, K)
            token_mask = mask_expert.any(dim=1)
            if token_mask.sum() == 0:
                continue

            x_selected = x_flat[token_mask]
            out = self.experts[expert_id](x_selected)

            # compute final output using weighted contribution from positions in top-k
            for k in range(K):
                mask_k = token_mask & (topk_idx[:, k] == expert_id)
                if mask_k.any():
                    expert_outputs[mask_k] += topk_weights[mask_k, k].unsqueeze(-1) * out[mask_k[token_mask]]

        return expert_outputs.view(B, T, self.output_dim)

# attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, heads_num, has_query_weight=True):
        super().__init__()
        self.input_dim = input_dim
        self.heads_num = heads_num
        self.head_dim = input_dim // heads_num

        if self.head_dim * heads_num != input_dim:
            raise ValueError(
                f"Embedding size needs to be divisible by heads_num {heads_num}"
            )

        self.getvalues = nn.Linear(input_dim, input_dim, bias=False)
        self.getkeys = nn.Linear(input_dim, input_dim, bias=False)

        if has_query_weight:
            self.getqueries = nn.Linear(input_dim, input_dim, bias=False)

        self.mh_l = nn.Linear(input_dim, input_dim, bias=False)
        self.has_query_weight = has_query_weight

    def forward(self, values_in, keys_in, query_in, mask_k=None, mask_q=None):

        # input size: (B, T, D)
        B = values_in.shape[0]
        T_value = values_in.shape[1]
        T_key = keys_in.shape[1]
        T_query = query_in.shape[1]

        values = self.getvalues(values_in)
        keys = self.getkeys(keys_in)
        if self.has_query_weight:
            queries = self.getqueries(query_in)
        else:
            queries = query_in

        values = values.reshape(B, T_value, self.heads_num, self.head_dim)
        keys = keys.reshape(B, T_key, self.heads_num, self.head_dim)
        queries = queries.reshape(B, T_query, self.heads_num, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        # mask_k (B, T_key, 1) and mask_q (B, T_query, 1)
        if mask_k is not None and mask_q is not None:
            mask_energy = mask_q & mask_k.squeeze(2).unsqueeze(1)  # (B, T_query, T_key)
            mask_energy = mask_energy.unsqueeze(1)  # (B, 1, T_query, T_key)
            energy = energy.masked_fill(mask_energy == False, float('-1e20'))
        elif mask_k is not None and mask_q is None:
            mask_energy = mask_k.squeeze(2).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_key)
            energy = energy.masked_fill(mask_energy == False, float('-1e20'))
        elif mask_k is None and mask_q is not None:
            mask_energy = mask_q.unsqueeze(1).unsqueeze(2)  # (B, 1, T_query, 1)
            energy = energy.masked_fill(mask_energy == False, float('-1e20'))

        attention = torch.softmax(energy / (self.input_dim ** (1 / 2)), dim=3)

        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values]).reshape(
            B, T_query, self.heads_num * self.head_dim
        )
        out = self.mh_l(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, heads, out_dim=None, num_experts=2, noise_std = 1.0 ,has_query_weight=True):
        super().__init__()
        # self.input_dim = input_dim
        if out_dim is None:
            out_dim = input_dim
        self.out_dim = out_dim
        self.norm0 = nn.LayerNorm(input_dim)
        self.attention = Attention(input_dim, heads, has_query_weight)
        self.norm1 = nn.LayerNorm(input_dim)
        self.MoE = MoE(input_dim, out_dim,num_experts, noise_std)

    def forward(self,key, query=None, value=None,  mask_k=None, mask_q=None):

        key_norm = self.norm0(key)
        if value is None:
            value_norm = key_norm
        else:
            value_norm = self.norm0(value)

        if query is None:
            query_norm = key_norm
        else:
            query_norm = self.norm0(query)

        if mask_k is not None and query is None:
            mask_q = mask_k
        attention = self.attention(value_norm, key_norm, query_norm, mask_k, mask_q)

        if not mask_q is None:
            attention = attention.masked_fill(mask_q == False, 0)

        # residual connection
        if query is None:
            attention = attention + key
        else:
            attention = attention + query

        out = self.norm1(attention)
        out = self.MoE(out)
        if self.out_dim == attention.shape[-1]:
            out = out + attention # residual connection

        if mask_q is not None:
            out = out.masked_fill(mask_q == False, 0)
        return out

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_index):
        super().__init__()
        self.pad_index = pad_index
        self.vocab_emb = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.float_proj = nn.Linear(1, embed_size)  # Project float to embed space
        self.number_index = vocab_size  # Used only to mask for floats

    def forward(self, inputwords):
        negative_mask = (inputwords < 0)
        mask = (inputwords != self.pad_index)

        inputwords_clipped = inputwords.clone()
        inputwords_clipped[negative_mask] = 0  # dummy index for floats

        word_emb = self.vocab_emb(inputwords_clipped.long())  # (B, T, D)

        if negative_mask.any():
            float_values = inputwords[negative_mask].unsqueeze(-1)  # (N, 1)
            float_emb = self.float_proj(float_values)               # (N, D)
            word_emb[negative_mask] = float_emb

        return word_emb, mask


class WordPosEncoding(nn.Module):
    def __init__(self, embed_size, maxwords):
        super().__init__()
        self.embed_size = embed_size

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(maxwords, embed_size)
        for pos in range(maxwords):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inputwords):
        # make embeddings relatively larger
        input_embedding = inputwords * math.sqrt(self.embed_size)
        # add constant to embedding
        input_embedding = input_embedding + self.pe[:, :inputwords.size(1), :]
        # input_embedding = input_embedding * mask.unsqueeze(2)
        return input_embedding


class PropEncoder(nn.Module):
    def __init__(self, embedsize, headnum, vocab_size, pad_index, maxwords=20, moe_noise_std = 1.0):
        super().__init__()

        self.embedding = WordEmbedding(vocab_size, embedsize, pad_index)
        self.wposencoding = WordPosEncoding(embedsize, maxwords)

        # word encoding
        self.transformer_w0 = TransformerLayer(embedsize, headnum, noise_std=moe_noise_std)
        self.token_w0 = nn.Parameter(torch.ones(1, 1, embedsize))
        self.transformer_w1 = TransformerLayer(embedsize, headnum,has_query_weight=False, noise_std=moe_noise_std)

        # sentence encoding
        # self.transformer_s0 = TransformerLayer(embedsize, headnum, noise_std=moe_noise_std)

    def forward(self, inputwords):
        # word embedding
        embeddedwords, mask = self.embedding(inputwords)

        # get the shape of the input
        B = embeddedwords.shape[0]
        T_s = embeddedwords.shape[1]
        T_w = embeddedwords.shape[2]
        embeddedwords = embeddedwords.reshape(B * T_s, T_w, -1)

        mask_kw = mask.reshape(B * T_s, T_w, -1)  # key mask for word
        mask_ks = (mask.sum(dim=2) != 0).unsqueeze(2)  # key mask for sentence
        mask_qw = mask_ks.reshape(B * T_s, 1, 1)  # query mask for word

        # position encoding for word
        embeddedwords = self.wposencoding(embeddedwords)

        # word encoding transformer layer 0
        outtrans_w0 = self.transformer_w0(embeddedwords, None, None, mask_kw)
        # word encoding transformer layer 1
        token_w0 = self.token_w0.repeat(B * T_s, 1, 1)
        outtrans_w1 = self.transformer_w1(outtrans_w0, token_w0, None, mask_kw, mask_qw)
        outtextembedding = outtrans_w1.reshape(B, T_s, -1)

        # sentence encoding transformer layer 0
        # outtextembedding = self.transformer_s0(outtextembedding, None, None, mask_ks, None) + outtextembedding

        return outtextembedding

class SpecEncoder(nn.Module):
    def __init__(self, bandwidth, bandnum, headnum, expertnum, moe_noise_std=1.0):
        super().__init__()

        self.bandwidth = bandwidth
        self.bandnum = bandnum

        # Embedding layers
        self.specposencoding = nn.Parameter(torch.rand(1, bandnum, bandwidth))
        self.transformer0 = TransformerLayer(bandwidth, headnum, bandwidth, expertnum, moe_noise_std)

    def forward(self, spectra):
        # spectra shape: (batch, bandnum, bandwidth)
        # positional encoding
        specposencoding = self.specposencoding.repeat(spectra.shape[0], 1, 1)

        specembedding = spectra + specposencoding  # shape: (batch, bandnum, bandwidth)

        return specembedding

class AutoRegression(nn.Module):
    def __init__(self, headnum, expertnum, vocab_size, pad_index, bandwidth=50, bandnum=43, maxwords = 20, moe_noise_std = 1.0, hidden_dim=96):
        super().__init__()
        self.projectin = nn.Linear(bandwidth, hidden_dim, bias=False)
        self.specencoder = SpecEncoder(hidden_dim, bandnum, headnum, expertnum, moe_noise_std)
        self.propencoder = PropEncoder(hidden_dim,headnum, vocab_size, pad_index, maxwords,moe_noise_std)

        self.transformer0 = TransformerLayer(hidden_dim, headnum, hidden_dim, expertnum, moe_noise_std)
        self.transformer1 = TransformerLayer(hidden_dim, headnum, hidden_dim, expertnum, moe_noise_std)
        self.transformer2 = TransformerLayer(hidden_dim, headnum, hidden_dim, expertnum, moe_noise_std)
        self.transformer3 = TransformerLayer(hidden_dim, headnum, hidden_dim, expertnum, moe_noise_std)
        self.transformer4 = TransformerLayer(hidden_dim, headnum, hidden_dim, expertnum, moe_noise_std)
        self.transformer5 = TransformerLayer(hidden_dim, headnum, bandwidth, expertnum, moe_noise_std)
        self.bandwidth = bandwidth

    def forward(self, spectra, propdata):
        # get mask
        mask = (spectra == 0).any(dim=2)  # Shape: (batch, bandnum)
        mask = mask.unsqueeze(2)  # Shape: (batch, bandnum, 1)

        # spectra shape: (batch, bandnum, bandwidth)
        spectra_input = self.projectin(spectra)  # Project spectra to hidden dimension (batch, bandnum, hidden_dim)

        # encode spectra
        specembedding = self.specencoder(spectra_input)

        # encode properties
        propembedding = self.propencoder(propdata)

        # Concatenate the embeddings
        allembedding = torch.cat((specembedding, propembedding), dim=1)
        mask_embedding = (allembedding != 0).any(dim=2)  # Shape: (batch, bandnum+propdata.shape[1], 1)
        mask_embedding = mask_embedding.unsqueeze(2) # Shape: (batch, bandnum+propdata.shape[1], 1)

        out_trans0 = self.transformer0(allembedding, specembedding, None, mask_embedding) + specembedding
        allembedding = torch.cat((out_trans0, propembedding), dim=1)
        out_trans1 = self.transformer1(allembedding, out_trans0, None, mask_embedding) + out_trans0
        allembedding = torch.cat((out_trans1, propembedding), dim=1)
        out_trans2 = self.transformer2(allembedding, out_trans1, None, mask_embedding) + out_trans1
        allembedding = torch.cat((out_trans2, propembedding), dim=1)
        out_trans3 = self.transformer3(allembedding, out_trans2, None, mask_embedding) + out_trans2
        allembedding = torch.cat((out_trans3, propembedding), dim=1)
        out_trans4 = self.transformer4(allembedding, out_trans3, None, mask_embedding) + out_trans3
        allembedding = torch.cat((out_trans4, propembedding), dim=1)
        out_spectra = self.transformer5(allembedding, out_trans4, None, mask_embedding)
        return out_spectra, mask
    
    def pad(self, spectra, propdata, smoothing = True):
        pad_spectra, mask_pad = self.forward(spectra, propdata)
        device_input = spectra.device
        mask_pad = mask_pad.repeat(1,1,self.bandwidth)
        if smoothing:
            windowsize = 7
            smoothstep = 5
            pad = windowsize//2
            conv_kernel = torch.ones(1, 1, windowsize) / windowsize
            conv_kernel = conv_kernel.to(device_input)
            pad_spectra[~mask_pad] = spectra[~mask_pad]
            spectra_to_smooth = torch.cat((pad_spectra[:,:-1,-smoothstep-pad:],pad_spectra[:,1:,:smoothstep + pad]),dim=2)
            spectra_smoothed = F.conv1d(spectra_to_smooth, conv_kernel.expand(spectra_to_smooth.size(1), -1, -1), groups=spectra_to_smooth.size(1))
            pad_spectra[:,:-1,-smoothstep:] = spectra_smoothed[:,:,:smoothstep]
            pad_spectra[:,1:,:smoothstep] = spectra_smoothed[:,:,-smoothstep:]

        pad_spectra[spectra>0] = spectra[spectra>0]
        return pad_spectra


