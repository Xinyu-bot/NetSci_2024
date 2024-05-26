import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_channels=55,
        hidden_channels_1=32,
        hidden_channels_2=16,
        output_channels=2,
        dropout=0.5,
    ):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_channels_1)
        self.fc2 = nn.Linear(hidden_channels_1, hidden_channels_2)
        self.fc3 = nn.Linear(hidden_channels_2, output_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        self.output_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.output_softmax(x)
