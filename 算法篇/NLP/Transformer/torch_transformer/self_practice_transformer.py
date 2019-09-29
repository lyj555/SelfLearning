# -*- coding: utf-8 -*-

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _padding_mask(seq_k, seq_q, pad_value=0):
    len_q = seq_q.size(1)  # seq_k和seq_q的形状都是[B,L]
    pad_mask = seq_k.eq(pad_value)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def _sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class PositionEncoding(nn.Module):
    def __init__(self, model_dim, max_len, device):
        super(PositionEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / model_dim)) for i in range(model_dim)]
                                for pos in range(max_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    def forward(self):
        out = nn.Parameter(self.pe, requires_grad=False).to(self.device)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None, attn_mask=None):
        """
        scale dot production attention(self attention)
        :param Q: query, [batch_size*num_head, max_len, dim_head]
        :param K: key, [batch_size*num_head, max_len, dim_head]
        :param V: value, [batch_size*num_head, max_len, dim_head]
        :param scale: scale factor, sqrt(dim_K), default None.
        :param attn_mask: attention mask, default None
        :return: attention value, softmax(QK^T)V, [batch_size*num_head, max_len, dim_head]
        """
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # [batch_size*num_head, max_len, max_len],
                                                         # 可以理解为做了num_head次的QK^T
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert model_dim % num_head == 0
        self.dim_head = model_dim // self.num_head

        self.fc_Q = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_K = nn.Linear(model_dim, num_head * self.dim_head)
        self.fc_V = nn.Linear(model_dim, num_head * self.dim_head)
        self.attention = ScaledDotProductAttention()

        self.fc = nn.Linear(self.num_head * self.dim_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        multi head attention
        :param query: (batch_size, max_len, model_dim)
        :param key:
        :param value:
        :param attn_mask: attention mask
        :return: (batch_size, max_len, model_dim)
        """
        batch_size = query.size(0)

        # linear projection
        Q = self.fc_Q(query)
        K = self.fc_K(key)
        V = self.fc_V(value)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_head, 1, 1)

        scale = K.size(-1) ** -0.5  # 缩放因子
        context, attention = self.attention(Q, K, V, scale, attn_mask)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)  # change back
        out = self.fc(context)
        out = self.dropout(out)
        out = out + query  # residual connection
        out = self.layer_norm(out)  # layer normalization
        return out, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        """
        position wise feed forward
        :param x: the output of multi head attention, (batch_size, max_len, model_dim)
        :return: (batch_size, max_len, model_dim)
        """
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # residual connection
        out = self.layer_norm(out)  # layer normalization
        return out


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_head, hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, hidden, dropout)

    def forward(self, x, attn_mask=None):
        """
        only one encoder layer
        :param x: inputs, (batch_size, max_len, model_dim)
        :param attn_mask: attention mask
        :return: (batch_size, max_len, model_dim)
        """
        out, attention = self.attention(query=x, key=x, value=x, attn_mask=attn_mask)
        out = self.feed_forward(out)
        return out, attention


class Encoders(nn.Module):
    def __init__(self, use_pretrain_embedding, pretrained_embedding, vocab_size, model_dim, max_len,
                 num_head, hidden, dropout, num_encoder, device):
        super(Encoders, self).__init__()
        if use_pretrain_embedding is not None:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        else:
            self.word_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=vocab_size - 1)

        self.position_embedding = PositionEncoding(model_dim, max_len, device)
        self.encoder = EncoderLayer(model_dim, num_head, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(num_encoder)])

    def forward(self, x):
        """
        transformer model
        :param x: input, (word_line, seq_len), word_line是index之后，seq_len是截断后的真是长度。形如([[20, 34, 90], [87, 34, 23], [23, 67, 89]], [3, 3, 2])
        :return:
        """
        out = self.word_embedding(x) + self.position_embedding
        self_attention_padding_mask = _padding_mask(out, out)

        attentions = []
        for encoder in self.encoders:
            out, attention = encoder(out, self_attention_padding_mask)
            attentions.append(attention)
        return out, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_head, dropout, max_len, device):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_head, dropout)
        self.feed_forward = PositionEncoding(model_dim, max_len, device)

    def forward(self, dec_input, enc_output, self_attn_mask=None, context_attn_mask=None):
        """
        only one decoder layer
        :param dec_input:
        :param enc_output:
        :param self_attn_mask:
        :param context_attn_mask:
        :return:
        """
        dec_output, self_attention = self.attention(query=dec_input, key=dec_input,
                                                    value=dec_input, attn_mask=self_attn_mask)
        # context attention
        dec_output, context_attention = self.attention(key=enc_output, value=enc_output,
                                                       query=dec_output, attn_mask=context_attn_mask)
        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attention, context_attention


class Decoders(nn.Module):
    def __init__(self, use_pretrain_embedding, pretrained_embedding, vocab_size, model_dim, max_len,
                 num_head, hidden, dropout, num_decoder, device):
        super(Decoders, self).__init__()

        if use_pretrain_embedding is not None:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        else:
            self.word_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=vocab_size - 1)

        self.position_embedding = PositionEncoding(model_dim, max_len, device)

        self.decoder = DecoderLayer(model_dim, num_head, dropout, max_len, device)
        self.decoder_layers = nn.ModuleList(
            [copy.deepcopy(self.decoder) for _ in range(num_decoder)])

    def forward(self, dec_input, enc_output, context_attn_mask=None):
        out = self.word_embedding(dec_input) + self.position_embedding

        self_attention_padding_mask = _padding_mask(dec_input, dec_input)
        seq_mask = _sequence_mask(dec_input)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attention, context_attention = decoder(out, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attention)
            context_attentions.append(context_attention)

        return out, self_attentions, context_attentions


class Transformer(nn.Module):

    def __init__(self, use_pretrain_embedding, pretrained_embedding, model_dim, src_vocab_size, src_max_len,
                 num_head, hidden, num_layers, dropout, device, tgt_vocab_size, tgt_max_len):
        super(Transformer, self).__init__()
        self.encoder = Encoders(use_pretrain_embedding, pretrained_embedding, src_vocab_size, model_dim,
                                src_max_len, num_head, hidden, dropout, num_layers, device)
        self.decoder = Decoders(use_pretrain_embedding, pretrained_embedding, tgt_vocab_size, model_dim, tgt_max_len,
                                num_head, hidden, dropout, num_layers, device)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, tgt_seq):
        context_attn_mask = _padding_mask(tgt_seq, src_seq)

        output, enc_self_attn_list = self.encoder(src_seq)
        output, dec_self_attn, ctx_attn = self.decoder(dec_input=tgt_seq, enc_output=output,
                                                       context_attn_mask=context_attn_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output, enc_self_attn_list, dec_self_attn, ctx_attn

