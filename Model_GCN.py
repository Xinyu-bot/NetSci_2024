import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        input_channels=55,
        hidden_channels_1=32,
        hidden_channels_2=16,
        output_channels=2,
        dropout=0.5,
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, output_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        self.output_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.activation(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return self.output_softmax(x)
