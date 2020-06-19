import torch
from torch import nn
from modules import *
from utils import get_glimpse
import math


class GDRAM(nn.Module):
    def __init__(self, device=None, dataset=None, Fast=False):
        super(GDRAM, self).__init__()

        self.glimpse_size = 12
        self.num_scales = 4

        self.img_size = 128

        self.class_num = 10

        if dataset == 'cifar100':
            self.class_num = 100


        self.normalized_glimpse_size = self.glimpse_size/(self.img_size/2)

        self.glimpse_net = GlimpseNetwork(3*self.num_scales,self.glimpse_size,2,128,128)

        self.rnn1 = GlimpseLSTMCoreNetwork(128,128)
        self.rnn2 = LocationLSTMCoreNetwork(128,128,self.glimpse_size)

        self.class_net = ActionNetwork(128, self.class_num)
        self.emission_net = EmissionNetwork(128)

        self.baseline_net = BaselineNetwork(128*2,1)

        self.num_glimpses = 8
        self.location_size = 2

        self.device = device

        self.fast = Fast

    def forward(self, x):

        batch_size = x.size(0)

        hidden1, cell_state1 = self.rnn1.init_hidden(batch_size)
        hidden1 = hidden1.to(self.device)
        cell_state1 = cell_state1.to(self.device)


        hidden2, cell_state2 = self.rnn2.init_hidden(x, batch_size)

        hidden2 = hidden2.to(self.device)
        cell_state2 = cell_state2.to(self.device)

        #location = torch.zeros(batch_size,2).to(self.device)
        std = (torch.ones(batch_size,2)*(math.exp(-1/2))).to(self.device)

        location, std, log_prob = self.emission_net(hidden2)
        location = torch.clamp(location, min=-1 + self.normalized_glimpse_size / 2,
                              max=1 - self.normalized_glimpse_size / 2)

        location_log_probs = torch.empty(batch_size, self.num_glimpses).to(self.device)
        locations = torch.empty(batch_size, self.num_glimpses, self.location_size).to(self.device)
        baselines = torch.empty(batch_size, self.num_glimpses).to(self.device)
        weights = torch.empty(batch_size, self.num_glimpses).to(self.device)

        weight = torch.ones(batch_size).to(self.device)

        action_logits = 0
        weight_sum = 0


        for i in range(self.num_glimpses):



            locations[:, i] = location

            location_log_probs[:, i] = log_prob

            glimpse = get_glimpse(x, location.detach(), self.glimpse_size, self.num_scales, device=self.device).to(self.device)
            glimpse_feature = self.glimpse_net(glimpse, location)

            hidden1, cell_state1 = self.rnn1(glimpse_feature, (hidden1, cell_state1))
            hidden2, cell_state2 = self.rnn2(hidden1, (hidden2, cell_state2))

            loc_diff, std, log_prob = self.emission_net(hidden2)
            loc_diff *= (self.normalized_glimpse_size/2 * 2**(self.num_scales - 1))
            new_location = location.detach() + loc_diff
            new_location = torch.clamp(new_location, min = -1 + self.normalized_glimpse_size/2 , max= 1 - self.normalized_glimpse_size/2)


            location = new_location

            hidden = torch.cat((hidden1, hidden2), dim=1)
            baseline = self.baseline_net(hidden)

            #location_log_probs[:, i] = log_prob
            baselines[:, i] = baseline.squeeze()

            weight = weight.unsqueeze(1)
            action_logit = self.class_net(hidden1)

            action_logits += weight*action_logit

            weights[:,i] = weight.squeeze()

            weight_sum += weight

            if (not self.training and i>1) and self.fast:
                if weights[0,-1]<0.5 and weights[0,-2]<0.5:
                    break

            std = torch.mean(std, dim=1)
            normalized_std = (std-math.exp(-1/2))/(math.exp(1/2)-math.exp(-1/2))
            weight = 1 - normalized_std

        action_logits /= weight_sum

        return  action_logits, locations, location_log_probs, baselines, weights

if __name__ == '__main__':
    model = DRAM(uniform=True, cpu=False).cuda()
    model.eval()
    input = torch.ones((1,3,32,32)).cuda()

    action_logits, locations, location_log, baseline, weights = model(input)

    print(action_logits, locations.shape, location_log.shape, baseline.shape, weights.shape)






