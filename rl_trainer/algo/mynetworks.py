from pathlib import Path
import sys
from collections import OrderedDict
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *

HIDDEN_SIZE = 256
CNN_OUT1 = 64
CNN_OUT2 = 128


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim 
        self.act_dim = act_dim
        self.num_agents = num_agents
        #self.batch_size = batch_size
        self.args = args
        ### rewrite the code for obs encoding
        ## input = [batch, 2, 10, 20]
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels= 2,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('conv2', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('conv3', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('maxpooling1', nn.MaxPool2d(
                      kernel_size=(2,2),
                      #stride=2,
            )),
        ]))

        self.conv_block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=CNN_OUT2,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=CNN_OUT2,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu2', nn.ReLU()),
            ('maxpooling1', nn.MaxPool2d(
                      kernel_size=(2,2),
                      #stride=2,
            )),
        ]))

        #conv3 = nn.Conv2d(in_channels=CNN_OUT2, out_channels=128, )
        #self.flatten = .reshape(-1, 1, 2*5*CNN_OUT2)

        self.dense1 = mlp([CNN_OUT2*5*2, 256, 256])
        self.lstm = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dense2 = nn.Linear(256*2, act_dim)
        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]
        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]
        

    def forward(self, obs):
        out = self.conv_block1(obs)
        #print('------outsize---------: {}'.format(obs.size()))
        out = self.conv_block2(out)
        #flatten
        out = out.reshape(-1, 1, 2*5*CNN_OUT2)
        out = self.dense1(out)
        out = self.lstm(out)
        out = self.dense2(out)
        #print('------outsize2---------: {}'.format(out.size()))
        #res = [out for i in range(3)]
        #res = torch.cat(res).reshape(-1, 3, self.act_dim)
        out = torch.broadcast_to(out, [-1, 3, self.act_dim])
        # res = torch.Tensor(size=(out.size()[0],3,4))
        # for i in range(3):
        #     res[i][:] = out
        #print('-------res-------: {}'.format(res) )
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim + 2*act_dim, HIDDEN_SIZE]

        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels= 2,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('conv2', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('conv3', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT1,
                      kernel_size= (3,3),
                      padding='same')),
            ('maxpooling1', nn.MaxPool2d(
                      kernel_size=(2,2),
                      #stride=2,
            )),
        ]))

        self.conv_block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=CNN_OUT1,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=CNN_OUT2,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(in_channels=CNN_OUT2,
                      out_channels= CNN_OUT2,
                      kernel_size= (3,3),
                      padding='same')),
            #('relu2', nn.ReLU()),
            ('maxpooling1', nn.MaxPool2d(
                      kernel_size=(2,2),
                      #stride=2,
            )),
        ]))

        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, 1]
        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.dense1 = mlp([CNN_OUT2*5*2+act_dim, 256, 256])
        self.lstm = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dense2 = nn.Linear(256*2, 1)

    def forward(self, obs_batch, action_batch):
        # action_batch = [batch_size, 3, 4]
        out = self.conv_block1(obs_batch)
        out = self.conv_block2(out)
        #flatten
        out = out.reshape(-1, 1, 2*5*CNN_OUT2)
        #out = [out for i in range(3)]
        #out = torch.cat(out).reshape(-1, 3, 2*5*CNN_OUT2)
        out = torch.broadcast_to(out, [-1, 3, 2*5*CNN_OUT2])
        #out = self.dense1(out)
        #print('-----action_size-----', action_batch.size(), out.size())
        action_batch = action_batch.reshape(-1, 3, self.act_dim)
        #print('-----action_size2-----', action_batch.size(), out.size())
        out = torch.cat((out, action_batch), dim=-1)
        out = self.dense1(out)
        out = self.lstm(out)
        out = self.dense2(out)        
        # if self.args.algo == "bicnet":
        #     out = self.comm_net(out)
        print('-----critic------', out.size())
        #out = self.post_dense(out)
        return out


class LSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data, ):
        output, (_, _) = self.lstm(data)
        return output
