import torch
import random
import numpy as np
from torch import nn
from dino import Game, num_actions, action_list
from pathlib import Path

file_dir = Path(__file__).parent
model_weights_file = "model_weights.pth"
model_weights_dir = file_dir / "trained_model"
model_weights_path = model_weights_dir / model_weights_file

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-2
num_episodes = int(1e4)
maximum_episode_length = int(1e10)
memory_buffer_capacity = int(1e3)
discount_factor = 1
update_per_timesteps = 10
batch_size = 100
greedy_factor = 0.3
save_model_per_episodes = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.neural_network = nn.Sequential(
            # Complex nn
            # nn.Conv2d(1, 16, 4, 2),
            # nn.MaxPool2d(2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(16, 32, 4, 2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 64, 2, 1),
            # nn.ReLU(inplace=True),
            # nn.Flatten(),
            # nn.Linear(768, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, num_actions),
            # Simple nn
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1344, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )
    
    # Conv2d input shape: (current_batch_size, channels_in, height_in, width_in)
    # here we use x with shape (current_batch_size, 1, frame_shape[0], frame_shape[1])
    def forward(self, x):
        logits = self.neural_network(x)
        return logits

# init weights and bias for nn layers
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        torch.nn.init.constant_(m.bias, 0.01)

# get frame input of shape (1, 1, frame_shape[0], frame_shape[1]) for model
def get_frame_input(frame):
    frame_input = torch.from_numpy(frame).type(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return frame_input

# update reward for previous memory when crashed because of the delay of feedback from environment
def revise_memory(memory_buffer, game_over):
    revise_steps = 10
    revise_reward = -5
    if game_over:
        for i in range(max(len(memory_buffer) - revise_steps, 0), len(memory_buffer)):
            memory_buffer[i][2] = revise_reward

def train():
    model = Model().to(device)
    game = Game()

    # load pretrained model
    if (model_weights_path.exists()):
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.apply(init_weights)

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
                action = torch.argmax(output).to("cpu").numpy().item()
            reward, next_state, game_over = game.take_action(action)
            revise_memory(memory_buffer, game_over)
            memory_buffer.append([state, action, reward, next_state, game_over])
            if len(memory_buffer) > memory_buffer_capacity:
                memory_buffer.pop(0)
            
            # train model
            if (t + 1) % update_per_timesteps == 0:
                batch = random.sample(memory_buffer, min(len(memory_buffer), batch_size))
                action_batch = [e[1] for e in batch]
                x_batch = torch.stack([get_frame_input(e[0]) for e in batch]).squeeze(1)
                y_batch = torch.tensor([
                    e[2] if e[4] \
                    else e[2] + discount_factor * torch.max(model(get_frame_input(e[3]))).detach().to("cpu").numpy() \
                    for e in batch
                ]).float().to(device)

                # Compute prediction and loss
                pred = model(x_batch)[torch.arange(len(action_batch)), action_batch].to(device)
                loss = loss_fn(pred, y_batch)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(x_batch)
                # print(y_batch)
                # print(pred)
                print(f"episode: {i}, step: {t}, loss: {loss}")

            if game_over or t == maximum_episode_length - 1:
                game.restart()
                break

            state = game.get_frame()
        
        # save model
        if (i + 1) % save_model_per_episodes == 0:
            print(f"save model on episode: {i}")
            torch.save(model.state_dict(), model_weights_path)
    
    game.close()

def test():
    model = Model().to(device)
    game = Game()

    model.load_state_dict(torch.load(model_weights_path))

    game.open()
    game.start()

    while(True):
        state = game.get_frame()
        game.display(state)
        output = model(get_frame_input(state))
        action = torch.argmax(output).to("cpu").numpy()
        _, _, game_over = game.take_action(action)
        print(f"output: {output}, action: {action}")
        if game_over:
            game.restart()