import torch
from torch import nn
from dino import Game, num_actions

num_episodes = 1e4
maximum_episode_length = 1e10
memory_buffer_capacity = 1e3
discount_factor = 1
soft_update_factor = 0.8
update_per_timesteps = 10
batch_size = 32
greedy_factor = 0.3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.neural_network = nn.Sequential(
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(6720, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )
    
    def forward(self, x):
        logits = self.neural_network(x)
        return logits

def train():
    model = Model()
    game = Game()
    game.start()
    frame = game.get_frame()
    print(frame.shape)
    input = torch.randn(10, 1, 128, 256)
    # output = model.forward(torch.from_numpy(frame).type(torch.float32))
    output = model.forward(input)
    print(output)
    print(output.shape)

    # for i in range(num_episodes):
    #     pass

def test():
    pass