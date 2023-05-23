import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

import numpy as np
import math

from base import BaseSequentialModel
from utils.util import prepare_device

import os


class LSTM(BaseSequentialModel):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

    def forward(self, question, test, tag, correct, elapsed_question, elapsed_test, mask, interaction):
        X, batch_size = super().forward(
            test=test, question=question, tag=tag, correct=correct, mask=mask, interaction=interaction
        )
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(BaseSequentialModel):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, question, test, tag, correct, elapsed_question, elapsed_test, mask, interaction):
        X, batch_size = super().forward(
            test=test, question=question, tag=tag, correct=correct, mask=mask, interaction=interaction
        )

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(BaseSequentialModel):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        n_tests: int,
        n_questions: int,
        n_tags: int,
        n_heads: int,
        drop_out: float,
        max_seq_len: float,
        **kwargs
    ):
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, question, test, tag, correct, elapsed_question, elapsed_test, mask, interaction):
        X, batch_size = super().forward(
            test=test, question=question, tag=tag, correct=correct, mask=mask, interaction=interaction
        )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


### Last Query Transformer RNN
class FeedForwardBlock(nn.Module):
    """
    out = Relu(M_out * w1 + b1) * w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQueryModel(BaseSequentialModel):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        drop_out: float,
        max_seq_len: int,
        n_layers: int,
        use_lstm: bool = True,
        **kwargs
    ):
        super(LastQueryModel, self).__init__(hidden_dim, n_layers, drop_out)

        # n_questions: 9455
        # n_test: 1538
        # n_tags: 913
        # elapsed_question: 14067
        # elapsed_test: 14706

        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.max_seq_len = max_seq_len
        self.device, _ = prepare_device(1)

        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=self.drop_out)
        self.ffn = FeedForwardBlock(hidden_dim)
        # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        # self.mask = None 
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=hidden_dim, 
                                hidden_size=hidden_dim, 
                                num_layers=n_layers, 
                                batch_first=True)
            
        # GRU
        # self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)
 
    def forward(self, input):
        embed, batch_size = super().forward(input)

        q = self.query(embed).permute(1, 0, 2)  # transpose 와 비슷한 역할, contiguous 텐서가 아니어도 작동함
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, attn_wt = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.layer_norm1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.layer_norm2(out)

        ## LSTM
        hidden = self.init_hidden(batch_size)
        
        if self.use_lstm:
            out, hidden = self.lstm(out, hidden)

        # GRU
        # out, hidden = self.gru(out, hidden[0])

        ## DNN
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        print(preds)        
        return preds