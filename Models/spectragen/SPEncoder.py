import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_index, number_index):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pad_index = pad_index
        self.number_index = number_index

        # Initialize the weight matrix of the embedding layer
        self.weight = nn.Parameter(torch.Tensor(vocab_size, embed_size))
        nn.init.xavier_uniform_(self.weight)

        # Set the weights of the padding token to all zeros
        self.weight.data[pad_index] = torch.zeros(embed_size)

    def forward(self, inputwords):
        negative_mask = (inputwords <= 0)
        mask = (inputwords != self.pad_index)
        input_num = inputwords * negative_mask
        newwordidx = inputwords
        newwordidx[negative_mask] = self.number_index
        newwordidx = newwordidx.long()
        embedded = F.embedding(newwordidx, self.weight, padding_idx=self.pad_index)
        embedded = embedded - input_num.unsqueeze(3) / 100
        return embedded, mask


class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, max_seq_len=100):
        super(PositionalEncoder, self).__init__()
        self.embed_size = embed_size

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, embed_size)
        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_embedding):
        # make embeddings relatively larger
        input_embedding = input_embedding * math.sqrt(self.embed_size)
        # add constant to embedding
        seq_len = input_embedding.size(2)
        input_embedding = input_embedding + self.pe[:, :, :seq_len]
        return input_embedding


class feedforward(nn.Module):
    def __init__(self, embed_size, output_size=None):
        super(feedforward, self).__init__()
        if output_size is None:
            output_size = embed_size
        self.linear = nn.Sequential(
            nn.Linear(embed_size, embed_size*2),
            QuickGELU(),
            nn.Linear(embed_size*2, output_size),
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
    def __init__(self, embed_size, heads, output_size=None, token_query=False,residual = True):
        super(transformerlayer, self).__init__()
        # self.embed_size = embed_size
        if output_size is None:
            output_size = embed_size
        self.residual = residual
        if output_size != embed_size:
            if self.residual:
                print('Warning: residual is not possible')
                self.residual = False
        self.attention = attention(embed_size, heads, token_query)
        self.norm0 = nn.LayerNorm(embed_size)
        self.feedforward = feedforward(embed_size, output_size)
        self.norm1 = nn.LayerNorm(output_size)

    def forward(self, value, key, query, mask_k=None, mask_q=None):
        attention = self.attention(value, key, query, mask_k, mask_q)

        attention = self.norm0(attention)
        if not mask_q is None:
            attention = attention.masked_fill(mask_q.unsqueeze(2) == False, 0)

        out = self.feedforward(attention + query)

        if self.residual:
            out = self.norm1(out + query)
        else:
            out = self.norm1(out)

        if mask_q is not None:
            out = out * mask_q.unsqueeze(2)

        return out


class SPEncod(nn.Module):
    def __init__(self, embedsize, headnum, vocab_size, pad_index, number_index, outsize=256, maxwords=16):
        super(SPEncod, self).__init__()

        self.embedding = Embedding(vocab_size, embedsize, pad_index, number_index)
        self.positionencoding = PositionalEncoder(embedsize, maxwords)

        # word encoding
        self.transformer_w0 = transformerlayer(embedsize, headnum)
        self.transformer_w1 = transformerlayer(embedsize, headnum)
        self.transformer_w2 = transformerlayer(embedsize, headnum)
        self.token_w0 = nn.Parameter(torch.rand(1, 1, embedsize))
        self.transformer_w3 = transformerlayer(embedsize, headnum, None, True, False)

        # sentence encoding
        self.transformer_s0 = transformerlayer(embedsize, headnum)
        self.transformer_s1 = transformerlayer(embedsize, headnum)
        self.transformer_s2 = transformerlayer(embedsize, headnum)
        self.transformer_s3 = transformerlayer(embedsize, headnum, outsize)

    def forward(self, inputwords):
        # word embedding
        embeddedwords, mask = self.embedding(inputwords)

        # get the shape of the input
        N = embeddedwords.shape[0]
        senten_len = embeddedwords.shape[1]
        word_len = embeddedwords.shape[2]

        embeddedwords = embeddedwords.reshape(N * senten_len, word_len, -1)

        mask_kw = mask.reshape(N * senten_len, word_len)  # key mask for word
        mask_ks = (mask.sum(dim=2) != 0)  # key mask for sentence
        mask_qw = mask_ks.reshape(N * senten_len, 1)  # query mask for word

        # position encoding for word
        embeddedwords = self.positionencoding(embeddedwords)

        # word encoding transformer layer 0
        outtrans_w0 = self.transformer_w0(embeddedwords, embeddedwords, embeddedwords, mask_kw, mask_kw)

        # word encoding transformer layer 1
        outtrans_w1 = self.transformer_w1(outtrans_w0, outtrans_w0, outtrans_w0, mask_kw, mask_kw)

        # word encoding transformer layer 2
        outtrans_w2 = self.transformer_w2(outtrans_w1, outtrans_w1, outtrans_w1, mask_kw, mask_kw)

        # word encoding transformer layer 3
        token_w0 = self.token_w0.repeat(N * senten_len, 1, 1)
        outtrans_w3 = self.transformer_w3(outtrans_w2, outtrans_w2, token_w0, mask_kw, mask_qw)
        outword = outtrans_w3.reshape(N, senten_len, -1)

        # sentence encoding transformer layer 0
        outtrans_s0 = self.transformer_s0(outword, outword, outword, mask_ks, mask_ks) + outword

        # sentence encoding transformer layer 1
        outtrans_s1 = self.transformer_s1(outtrans_s0, outtrans_s0, outtrans_s0, mask_ks, mask_ks)

        # sentence encoding transformer layer 2
        outtrans_s2 = self.transformer_s2(outtrans_s1, outtrans_s1, outtrans_s1, mask_ks, mask_ks)

        # sentence encoding transformer layer 5
        token_s0 = outtrans_s2.sum(dim=1) / mask_ks.sum(dim=1).unsqueeze(1)
        token_s0 = token_s0.unsqueeze(1)
        outtextembedding = self.transformer_s3(outtrans_s2, outtrans_s2, token_s0, mask_ks)

        return outtextembedding