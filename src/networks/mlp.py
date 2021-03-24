import torch.nn as nn

from base.base_net import BaseNet


class MLPNet(BaseNet):
    """A naive fully-connected backbone model for non-convolutional inputs
    """

    def __init__(self):
        super(MLPNet, self).__init__()

        self.rep_dim = 32
        self.features_e = 16
        self.input_size = 100

        self.lrelu = nn.LeakyReLU()

        self.layer1 = nn.Linear(self.input_size,
                                self.features_e * 4,
                                bias=False)
        self.layer2 = nn.Linear(self.features_e * 4,
                                self.features_e * 2,
                                bias=False)
        self.layer3 = nn.Linear(self.features_e * 2, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.layer3(x)
        return x


class MLPNetAutoencoder(BaseNet):
    def __init__(self):
        super(MLPNetAutoencoder, self).__init__()

        self.rep_dim = 32
        self.features_e = 16
        self.input_size = 100

        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # Encoder input_size -> rep_dim
        self.layer1 = nn.Linear(self.input_size,
                                self.features_e * 4,
                                bias=False)
        self.layer2 = nn.Linear(self.features_e * 4,
                                self.features_e * 2,
                                bias=False)
        self.layer3 = nn.Linear(self.features_e * 2, self.rep_dim, bias=False)

        # Decoder rep_dim -> input_size
        self.layer4 = nn.Linear(self.rep_dim, self.features_e * 2, bias=False)
        self.layer5 = nn.Linear(self.features_e * 2,
                                self.features_e * 4,
                                bias=False)
        self.layer6 = nn.Linear(self.features_e * 4,
                                self.input_size,
                                bias=False)

    def forward(self, x):
        # Encoder
        x = self.layer1(x)
        x = self.lrelu(x)
        x = self.layer2(x)
        x = self.lrelu(x)
        x = self.layer3(x)

        # Decoder
        x = self.layer4(x)
        x = self.lrelu(x)
        x = self.layer5(x)
        x = self.lrelu(x)
        x = self.layer6(x)
        x = self.sigmoid(x)
        return x
