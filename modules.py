import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import numpy as np

class GlimpseNetwork(nn.Module):

    def __init__(self, input_channel, glimpse_size, location_size, internal_size, output_size):
        super(GlimpseNetwork, self).__init__()

        self.fc_g = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_l = nn.Sequential(
            nn.Linear(location_size, internal_size),
            nn.ReLU())

        self.fc_gg = nn.Linear(glimpse_size//4 * glimpse_size//4 * 256, output_size)
        self.fc_lg = nn.Linear(internal_size, output_size)

    def forward(self, x, location):
        hg = self.fc_g(x).view(len(x), -1)
        hl = self.fc_l(location)

        output = F.relu(self.fc_gg(hg) * self.fc_lg(hl))

        return output



class CoreNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(CoreNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(
            input_size, hidden_size, nonlinearity='relu')

    def forward(self, g, prev_h):
        h = self.rnn_cell(g, prev_h)
        return h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class GRUCoreNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCoreNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(
            input_size, hidden_size)

    def forward(self, g, prev_h):
        h = self.rnn_cell(g, prev_h)
        return h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class GlimpseLSTMCoreNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GlimpseLSTMCoreNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(
            input_size, hidden_size)

    def forward(self, g, prev_h):
        h, c = self.lstm_cell(g, prev_h)
        return h, c

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)


class LocationLSTMCoreNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, glimpse_size):
        super(LocationLSTMCoreNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.glimpse_size = glimpse_size

        self.lstm_cell = nn.LSTMCell(
            input_size, hidden_size)

        self.context_net1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.context_net2 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(glimpse_size//4*glimpse_size//4*64,hidden_size)

    def forward(self, g, prev_h):
        h, c = self.lstm_cell(g, prev_h)
        return h, c

    def init_hidden(self, x, batch_size):
        x = F.interpolate(x, (self.glimpse_size,self.glimpse_size))

        h = self.fc(self.context_net2(self.context_net1(x)).view(batch_size,-1))
        c = torch.zeros((batch_size, self.hidden_size))

        return  h, c


class EmissionNetwork(nn.Module):

    def __init__(self, input_size, uniform=False, output_size=2, hidden=256):
        super(EmissionNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU())

        self.mu_net = nn.Sequential(
            nn.Linear(hidden, output_size),
            nn.Tanh()
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(hidden, output_size),
            nn.Tanh()
        )

        self.unifrom = uniform

    def forward(self, x):

        z = self.fc(x.detach())
        mu = self.mu_net(z)

        logvar = self.logvar_net(z)
        std = torch.exp(logvar*0.5)

        if self.training:

            #distribution = torch.distributions.Normal(mu, std)
            distribution = torch.distributions.Normal(mu, std)
            output = torch.clamp(distribution.sample(), -1.0, 1.0)
            log_p = distribution.log_prob(output)
            log_p = torch.sum(log_p, dim=1)

        else:

            # output = F.tanh(mu)
            output = mu
            log_p = torch.ones(output.size(0))

        return output, std, log_p


class ActionNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden=256):
        super(ActionNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size)
        )

    def forward(self, x):
        logit = self.fc(x)

        return logit


class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        super(BaselineNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        output = torch.sigmoid(self.fc(x.detach()))
        return output
