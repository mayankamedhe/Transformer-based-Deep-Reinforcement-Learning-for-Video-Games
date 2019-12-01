import torch
import torch.nn as nn
import torch.nn.functional as F
from config_DQN import gamma, sequence_length, device
# torch.manual_seed(0)

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # print(1, x.shape)
        seq_length = x.size(1)
        if seq_length != sequence_length:
            x = torch.cat([x]*(sequence_length-seq_length+1), dim=1)
            # print('in', x.shape)
        x = x.view(-1, self.num_inputs * sequence_length)
        # print(2, x.shape)
        x = F.relu(self.fc1(x))
        # print(3, x.shape)
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.Tensor(batch.action).float().to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states)
        next_pred = target_net(next_states)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.l1_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]
