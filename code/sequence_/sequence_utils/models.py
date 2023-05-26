import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = self.args.device

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = self.args.hidden_dim, self.args.hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, intd)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, intd)
        self.embedding_elapsed_question = nn.Embedding(self.args.n_elapsed_question + 1, intd)
        # self.embedding_elapsed_test = nn.Embedding(self.args.n_elapsed_test + 1, intd)

        # Concatenated Embedding Projection
        self.comb_proj = nn.Linear(intd * 5, hd)
        self.fc = nn.Linear(hd, 1)
        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        # test, question, tag, _, elapsed_question, elapsed_test, mask, interaction, _ = input
        test, question, tag, _, elapsed_question, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_interaction = nn.Dropout(self.args.dropout)(embed_interaction)

        embed_test = self.embedding_test(test)
        embed_test = nn.Dropout(self.args.dropout)(embed_test)

        embed_question = self.embedding_question(question)
        embed_question = nn.Dropout(self.args.dropout)(embed_question)

        embed_tag = self.embedding_tag(tag)
        embed_tag = nn.Dropout(self.args.dropout)(embed_tag)

        embed_elapsed_question = self.embedding_elapsed_question(elapsed_question)
        embed_elapsed_question = nn.Dropout(self.args.dropout)(embed_elapsed_question)

        # embed_elapsed_test = self.embedding_elapsed_test(elapsed_test)
        # embed_elapsed_test = nn.Dropout(self.args.dropout)(embed_elapsed_test)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_elapsed_question,
                # embed_elapsed_test
            ],
            2,
        )

        X = self.comb_proj(embed)

        if (self.args.model == "bert") or (self.args.model == "lstm_attn"):
            return X, batch_size, mask

        return X, batch_size


class LSTM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.lstm = nn.LSTM(self.args.hidden_dim, self.args.hidden_dim, self.args.n_layers, batch_first=True)

    def forward(self, input):
        X, batch_size = super().forward(input=input)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)

        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATNN(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.lstm = nn.LSTM(self.args.hidden_dim, self.args.hidden_dim, self.args.n_layers, batch_first=True)

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_dropout_prob=self.args.dropout,
            attention_probs_dropout_prob=self.args.dropout,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, input):
        X, batch_size, mask = super().forward(input)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)

        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.args.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

    def forward(self, input):
        X, batch_size, mask = super().forward(input)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)

        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.scale * self.pe[: x.size(0), :]
        return self.dropout(x)


class Saint(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.args.hidden_dim // 3) * 4, self.args.hidden_dim)

        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.args.hidden_dim // 3) * 5, self.args.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.args.hidden_dim, self.args.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.args.hidden_dim, self.args.dropout, self.args.max_seq_len)

        self.transformer = nn.Transformer(
            d_model=self.args.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.args.hidden_dim,
            dropout=self.args.dropout,
            activation="relu",
        )

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1)).float()

        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, input):
        # test, question, tag, _, elapsed_question, elapsed_test, mask, interaction, _ = input
        test, question, tag, _, elapsed_question, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_elapsed_question = self.embedding_elapsed_question(elapsed_question)
        # embed_elapsed_test = self.embedding_elapsed_test(elapsed_test)

        embed_enc = torch.cat(
            [
                embed_test,
                embed_question,
                embed_tag,
                embed_elapsed_question,
                # embed_elapsed_test
            ],
            2,
        )

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed_elapsed_question = self.embedding_elapsed_question(elapsed_question)
        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat([embed_test, embed_question, embed_tag, embed_elapsed_question, embed_interaction], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(
            embed_enc, embed_dec, src_mask=self.enc_mask, tgt_mask=self.dec_mask, memory_mask=self.enc_dec_mask
        )

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQuery(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.args.hidden_dim)

        # Encoder
        self.query = nn.Linear(in_features=self.args.hidden_dim, out_features=self.args.hidden_dim)
        self.key = nn.Linear(in_features=self.args.hidden_dim, out_features=self.args.hidden_dim)
        self.value = nn.Linear(in_features=self.args.hidden_dim, out_features=self.args.hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.args.hidden_dim, num_heads=self.args.n_heads, dropout=self.args.dropout
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.args.hidden_dim)

        self.ln1 = nn.LayerNorm(self.args.hidden_dim)
        self.ln2 = nn.LayerNorm(self.args.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(self.args.hidden_dim, self.args.hidden_dim, self.args.n_layers, batch_first=True)

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def forward(self, input):
        X, batch_size = super().forward(input)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # X = X + embed_pos

        ####################### ENCODER #####################
        q = self.query(X).permute(1, 0, 2)

        q = self.query(X)[:, -1:, :].permute(1, 0, 2)

        k = self.key(X).permute(1, 0, 2)
        v = self.value(X).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = X + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = X + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class EncoderEmbedding(nn.Module):
    def __init__(self, args):
        self.args = args

        super(EncoderEmbedding, self).__init__()
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.args.hidden_dim)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.args.hidden_dim)

    def forward(self, question, tag, elapsed_question):
        embed_question = self.embedding_question(question)
        embed_question = nn.Dropout(self.args.dropout)(embed_question)

        embed_tag = self.embedding_tag(tag)
        embed_tag = nn.Dropout(self.args.dropout)(embed_tag)

        seq = torch.arange(self.args.max_seq_len, device=self.args.device).unsqueeze(0)

        embed_position = self.embedding_position(seq)
        embed_position = nn.Dropout(self.args.dropout)(embed_position)

        return embed_position + embed_tag + embed_question

class DecoderEmbedding(nn.Module):
    def __init__(self, args):
        self.args = args

        super(DecoderEmbedding, self).__init__()
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.args.hidden_dim)

    def forward(self, interaction):
        embed_interaction = self.embedding_interaction(interaction)
        embed_interaction = nn.Dropout(self.args.dropout)(embed_interaction)

        seq = torch.arange(self.args.max_seq_len, device=self.args.device).unsqueeze(0)
        
        embed_position = self.embedding_position(seq)
        embed_position = nn.Dropout(self.args.dropout)(embed_position)

        return embed_position + embed_interaction

class StackedNMultiHeadAttention(nn.Module):
    def __init__(self, args, n_multihead:int = 1):
        # n_stacks : # of encoder(decoder), default = 4
        # n_heads : # of encoder(decoder) heads, default = 8
        # n_multihead : # of multihead, default = 1(encoder), 2(decoder)
        
        self.args = args
        
        super(StackedNMultiHeadAttention, self).__init__()
        self.n_multihead = n_multihead 
        self.norm_layers = nn.LayerNorm(self.args.hidden_dim)

        # n_stacks has n_multiheads each
        self.multihead_layers = nn.ModuleList(self.args.n_layers * [nn.ModuleList(
                                                n_multihead * [nn.MultiheadAttention(
                                                                embed_dim = self.args.hidden_dim,
                                                                num_heads = self.args.n_heads,
                                                                dropout = self.args.dropout), ]), ])
        
        self.ffn = nn.ModuleList(self.args.n_layers * [Feed_Forward_block(self.args.hidden_dim)])
        self.mask = torch.triu(torch.ones(self.args.max_seq_len, self.args.max_seq_len),
                               diagonal=1).to(dtype=torch.bool)
        
    def forward(self, input_q, input_k, input_v, encoder_output=None, break_layer=None):
        for stack in range(self.args.n_layers):
            for multihead in range(self.n_multihead):
                norm_q = self.norm_layers(input_q)
                norm_k = self.norm_layers(input_k)
                norm_v = self.norm_layers(input_v)
                heads_output, attn_w = self.multihead_layers[stack][multihead](
                    query = norm_q.permute(1, 0, 2),
                    key = norm_k.permute(1, 0, 2),
                    value = norm_v.permute(1, 0, 2),
                    attn_mask = self.mask.to(self.args.device))
                
                heads_output = heads_output.permute(1, 0, 2)
                if encoder_output != None and multihead == break_layer:
                    assert break_layer <= multihead, "break layer should be less than multihead layers and positive integer"
                    input_k = input_v = encoder_output
                    input_q = input_q + heads_output
                else:
                    input_q = input_q + heads_output
                    input_k = input_k + heads_output
                    input_v = input_v + heads_output

            last_norm = self.norm_layers(heads_output)
            ffn_output = self.ffn[stack](last_norm)
            ffn_output = ffn_output + heads_output
        # after loops = input_q = input_k = input_v
        return ffn_output

class SaintPlus(nn.Module):
    def __init__(self, args):
        self.args = args

        super(SaintPlus, self).__init__()
        self.encoder_layer = StackedNMultiHeadAttention(args, n_multihead = 1)
        self.decoder_layer = StackedNMultiHeadAttention(args, n_multihead = 2)
        self.encoder_embedding = EncoderEmbedding(args)
        self.decoder_embedding = DecoderEmbedding(args)
        self.elapsed_time = nn.Linear(1, self.args.hidden_dim)
        
        self.fc = nn.Linear(self.args.hidden_dim, 1)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, elapsed_question, mask, interaction, _ = input
        batch_size = interaction.size(0)

        enc = self.encoder_embedding(question=question, tag=tag, elapsed_question=elapsed_question)
        dec = self.decoder_embedding(interaction=interaction)
        
        elapsed_question = elapsed_question.unsqueeze(-1).float()
        elapsed_question = self.elapsed_time(elapsed_question)

        dec = dec + elapsed_question

        # encoder
        encoder_output = self.encoder_layer(input_k=enc,
                                            input_q=enc,
                                            input_v=enc)
        # decoder
        decoder_output = self.decoder_layer(input_k=dec,
                                            input_q=dec,
                                            input_v=dec,
                                            encoder_output=encoder_output,
                                            break_layer=1)
        # fully-connected layer
        out = self.fc(decoder_output)
        preds = self.activation(out).view(batch_size, -1)

        return preds