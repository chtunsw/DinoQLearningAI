import torch
import random
import numpy as np
from torch import nn
from dino import Game, num_actions, action_list

learning_rate = 1e-2
num_episodes = int(1e4)
maximum_episode_length = int(1e10)
memory_buffer_capacity = int(1e3)
discount_factor = 1
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
            nn.Linear(256, num_actions),
        )
    
    # Conv2d input shape: (current_batch_size, channels_in, height_in, width_in)
    # here we use x with shape (current_batch_size, 1, frame_shape[0], frame_shape[1])
    def forward(self, x):
        logits = self.neural_network(x)
        return logits

# get frame input of shape (1, 1, frame_shape[0], frame_shape[1]) for model
def get_frame_input(frame):
    frame_input = torch.from_numpy(frame).type(torch.float32).unsqueeze(0).unsqueeze(0)
    return frame_input

def train():
    model = Model()
    game = Game()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    memory_buffer = []

    game.open()
    game.start()

    for i in range(num_episodes):
        state = game.get_frame()
        for t in range(maximum_episode_length):
            game.display(state)
            
            # take next action
            random_pick = random.uniform(0, 1) <= greedy_factor
            if random_pick:
                action = random.choice(action_list)
            else:
                output = model(get_frame_input(state))
                action = torch.argmax(output).numpy()
            reward, next_state, game_over = game.take_action(action)
            memory_buffer.append([state, action, reward, next_state, game_over])
            if len(memory_buffer) > memory_buffer_capacity:
                memory_buffer.pop(0)
            
            # train model
            if (t + 1) % update_per_timesteps == 0:
                batch = random.sample(memory_buffer, min(len(memory_buffer), batch_size))
                x_batch = torch.stack([get_frame_input(e[0]) for e in batch]).squeeze(1)
                y_batch = torch.tensor([
                    e[2] if e[4] \
                    else e[2] + discount_factor * torch.max(model(get_frame_input(e[3]))).detach().numpy() \
                    for e in batch
                ]).float()

                # Compute prediction and loss
                pred = torch.max(model(x_batch), dim=-1)[0]
                loss = loss_fn(pred, y_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(x_batch)
                # print(y_batch)
                # print(pred)
                # print(loss)

            if game_over:
                game.restart()
                break

            state = next_state
    
    game.close()

def test():
    pass