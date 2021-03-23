# =====Destiny======
# meta_model
import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, lib_sz, nclass):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(lib_sz+1, 128)
        self.lstm = nn.LSTM(128, 32, bidirectional=True)
        self.fc = nn.Linear(64, 2)
    
    def forward(self, x, text_lengths):
        x = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, text_lengths, 
                                                            batch_first=True)
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # self.lstm.flatten_parameters()
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # pylint: disable=no-member
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num directions]

        x = self.fc(hidden)
        return x