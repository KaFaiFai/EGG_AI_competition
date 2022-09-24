import torch
from torch import nn
from torchinfo import summary


class LinearNet(nn.Module):
    def __init__(self, num_class, num_channel, num_time_point):
        """
        num_class := 26
        num_channel := 24
        num_time_point := 801
        """

        super().__init__()
        self.num_class = num_class
        self.num_channel = num_channel
        self.num_time_point = num_time_point

        # channel-wise feed forward
        self.channel_forward = nn.Sequential(
            nn.Linear(num_time_point, num_time_point*2),
            nn.PReLU(num_channel),
            nn.Linear(num_time_point*2, num_time_point//4),
        )

        self.flatten = nn.Flatten()

        # skip softmax to use cross entropy as loss function 
        self.feed_forward = nn.Sequential(
            nn.Linear(num_time_point//4*num_channel,
                      num_time_point//4*num_channel*2),
            nn.PReLU(1),
            nn.Dropout(p=0.001),
            nn.Linear(num_time_point//4*num_channel*2, num_class),
            # nn.Softmax(dim=1),
        )

    def forward(self, x):
        # (B, num_channel, num_time_point) -> (B, num_class)
        x = self.channel_forward(x)
        x = self.flatten(x)
        x = self.feed_forward(x)
        return x

    def __repr__(self):
        return f"LinearNet({self.num_class})"


class LSTMNet(nn.Module):
    def __init__(self, num_class, num_channel, num_time_point):
        super().__init__()
        self.num_class = num_class
        self.num_channel = num_channel
        self.num_time_point = num_time_point

        self.lstm = nn.LSTM(num_time_point, num_time_point, 2, batch_first=True, dropout=0.05)

        self.flatten = nn.Flatten()

        self.feed_forward = nn.Sequential(
            nn.Linear(num_channel,
                      num_channel*2),
            nn.PReLU(1),
            nn.Linear(num_channel*2, num_class),
        )

    def forward(self, x):
        x = self.lstm(x)[0][:, :, -1]
        x = self.flatten(x)
        x = self.feed_forward(x)
        return x

    def __repr__(self):
        return f"LSTMNet({self.num_class})"

def test():
    net = LinearNet(26, 24, 801)
    batch_size = 16
    summary(net, (batch_size, 24, 801))


if __name__ == '__main__':
    test()
