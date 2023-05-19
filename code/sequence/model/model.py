import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

import numpy as np

from base import BaseSequentialModel

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
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
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


### LastQuery Transformer RNN
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


class LastQueryModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 1,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_elapsed_questions: int = 14066,
        n_elapsed_tests: int = 14705,
        n_heads: int = 1,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        use_lstm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.n_questions = len(np.load(os.path.join("asset/", "assessmentItemID_classes.npy")))
        self.n_tests = len(np.load(os.path.join("asset/", "testId_classes.npy")))
        self.n_tags = len(np.load(os.path.join("asset/", "KnowledgeTag_classes.npy")))
        self.n_elapsed_questions = len(np.load(os.path.join("asset/", "elapsed_question_classes.npy")))
        self.n_elapsed_tests = len(np.load(os.path.join("asset/", "elapsed_test_classes.npy")))

        # self.n_questions = n_questions
        # self.n_tests = n_tests
        # self.n_tags = n_tags
        # self.n_elapsed_questions = n_elapsed_question
        # self.n_elapsed_tests = n_elapsed_tests

        # self.embedding_correct = nn.Embedding(2, embedding_dim = hidden_dim)
        self.embedding_interaction = nn.Embedding(3, embedding_dim=hidden_dim)
        self.embedding_test = nn.Embedding(self.n_tests + 1, embedding_dim=hidden_dim)
        self.embedding_question = nn.Embedding(self.n_questions + 1, embedding_dim=hidden_dim)
        self.embedding_tag = nn.Embedding(self.n_tags + 1, embedding_dim=hidden_dim)
        self.embedding_elapsed_question = nn.Embedding(self.n_elapsed_questions + 1, embedding_dim=hidden_dim)
        self.embedding_elapsed_test = nn.Embedding(self.n_elapsed_tests + 1, embedding_dim=hidden_dim)

        self.drop_out = drop_out
        self.multi_en = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=self.drop_out)
        self.ffn_en = FeedForwardBlock(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers)

        self.out = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(
        self, question, test, tag, correct, elapsed_question, elapsed_test, mask, interaction, first_block=True
    ):
        first_block = True
        if first_block:
            embed_interaction = self.embedding_interaction(interaction.int())
            embed_interaction = nn.Dropout(self.drop_out)(embed_interaction)

            embed_question = self.embedding_question(question.int())
            embed_question = nn.Dropout(self.drop_out)(embed_question)

            embed_tag = self.embedding_tag(tag.int())
            embed_tag = nn.Dropout(self.drop_out)(embed_tag)

            embed_elapsed_question = self.embedding_elapsed_question(elapsed_question.int())
            embed_elapsed_question = nn.Dropout(self.drop_out)(embed_elapsed_question)

            embed_elapsed_test = self.embedding_elapsed_test(elapsed_test.int())
            embed_elapsed_test = nn.Dropout(self.drop_out)(embed_elapsed_test)

            embed_test = self.embedding_test(test.int())
            embed_test = nn.Dropout(self.drop_out)(embed_test)

            # out = embed_correct + embed_question + embed_tag + embed_elapsed_question + embed_elapsed_test
            out = (
                embed_interaction
                + embed_question
                + embed_test
                + embed_tag
                + embed_elapsed_question
                + embed_elapsed_test
            )

        else:
            out = embed_question

        out = out.permute(1, 0, 2)  # transpose 와 비슷한 역할, contiguous 텐서가 아니어도 작동함

        out = self.layer_norm1(out)  # Layer norm
        skip_out = out

        out, attn_wt = self.multi_en(out[-1:, :, :], out, out)  # Q, K, V
        out = out + skip_out  # skip connection

        if self.use_lstm:
            out, _ = self.lstm(out)
            out = out[-1:, :, :]

        # feed forward
        out = out.permute(1, 0, 2)  # (b, n, d)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out

        out = self.out(out)

        # return out.squeeze(-1), 0
        return out.squeeze(-1).view(-1)
