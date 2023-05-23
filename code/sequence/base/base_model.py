import torch
import torch.nn as nn

import numpy as np

import os

class BaseSequentialModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        drop_out: float
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_out = drop_out

        self.n_questions = len(np.load(os.path.join("asset/", "assessmentItemID_classes.npy")))
        self.n_tests = len(np.load(os.path.join("asset/", "testId_classes.npy")))
        self.n_tags = len(np.load(os.path.join("asset/", "KnowledgeTag_classes.npy")))
        self.n_elapsed_questions = len(np.load(os.path.join("asset/", "elapsed_question_classes.npy")))
        self.n_elapsed_tests = len(np.load(os.path.join("asset/", "elapsed_test_classes.npy")))

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, embedding_dim=intd)  # correct(1, 2) + padding(0)
        self.embedding_question = nn.Embedding(self.n_questions + 1, embedding_dim=intd)
        self.embedding_test = nn.Embedding(self.n_tests + 1, embedding_dim=intd)
        self.embedding_tag = nn.Embedding(self.n_tags + 1, embedding_dim=intd)
        self.embedding_elapsed_question = nn.Embedding(self.n_elapsed_questions + 1, embedding_dim=intd)
        self.embedding_elapsed_test = nn.Embedding(self.n_elapsed_tests + 1, embedding_dim=intd)

        # Concatentaed Embedding Projection
        self.comb_proj = nn.Linear(intd * 6, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)

    def forward(self, input):   # input: batch
        correct, question, test, tag , elapsed_question, elapsed_test, mask, interaction, index = input
        batch_size = interaction.size(0)
        
        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_interaction = nn.Dropout(self.drop_out)(embed_interaction)

        embed_question = self.embedding_question(question)
        embed_question = nn.Dropout(self.drop_out)(embed_question)
        
        embed_test = self.embedding_test(test)
        embed_test = nn.Dropout(self.drop_out)(embed_test)

        embed_tag = self.embedding_tag(tag)
        embed_tag = nn.Dropout(self.drop_out)(embed_tag)

        embed_elapsed_question = self.embedding_elapsed_question(elapsed_question)
        embed_elapsed_question = nn.Dropout(self.drop_out)(embed_elapsed_question)

        embed_elapsed_test = self.embedding_test(elapsed_test)
        embed_elapsed_test = nn.Dropout(self.drop_out)(embed_elapsed_test)

        embed = torch.cat(
            [
                embed_interaction,
                embed_question,
                embed_test,
                embed_tag,
                embed_elapsed_question,
                embed_elapsed_test,
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
