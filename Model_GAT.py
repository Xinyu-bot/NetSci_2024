import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        input_channels=55,
        hidden_channels_1=32,
        hidden_channels_2=16,
        output_channels=2,
        dropout=0.5,
        heads=1,
    ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            input_channels, hidden_channels_1, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels_1 * heads, hidden_channels_2, heads=heads, dropout=dropout
        )
        self.conv3 = GATConv(
            hidden_channels_2 * heads,
            output_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation = (
            nn.ReLU()
        )  # the activation function is ReLU here while in-layer activation is LeakyReLU in GATConv
        self.output_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.activation(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return self.output_softmax(x)
