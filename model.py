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

learning_rate = 1e-4
num_episodes = int(1e4)
maximum_episode_length = int(1e10)
memory_buffer_capacity = int(1e4)
discount_factor = 1
update_per_timesteps = 100
batch_size = 64
init_greedy_factor = 1e-1
final_greedy_factor = 1e-3
save_model_per_episodes = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.neural_network = nn.Sequential(
            # Complex nn
            nn.Conv2d(4, 16, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1568, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
            # Simple nn
            # nn.Conv2d(1, 16, 8, 4),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(16, 16, 4, 2),
            # nn.ReLU(inplace=True),
            # nn.Flatten(),
            # nn.Linear(1344, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, num_actions),
        )
    
    # Complex nn: take most recent 4 frames as input
    # Conv2d input shape: (current_batch_size, channels_in, height_in, width_in)
    # here we use x with shape (current_batch_size, 4, frame_shape[1], frame_shape[0])
    def forward(self, x):
        logits = self.neural_network(x)
        return logits

# init weights and bias for nn layers
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        torch.nn.init.constant_(m.bias, 0.01)

# get state input of shape (1, 4, frame_shape[1], frame_shape[0]) for model
def get_state_input(state):
    state_input = torch.from_numpy(state).type(torch.float32).unsqueeze(0)
    return state_input

def train():
    model = Model()
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

    total_steps = 0

    for i in range(num_episodes):
        for t in range(maximum_episode_length):
            total_steps += 1
            frame = game.get_frame()
            game.display(frame)

            # take most recent 4 frames as state
            if t == 0:
                state = np.stack((frame, frame, frame, frame))
            else:
                state = np.append(state[1:, :, :], np.expand_dims(frame, axis=0), axis=0)
            
            # take next action
            greedy_factor = init_greedy_factor - \
                (init_greedy_factor - final_greedy_factor) / num_episodes * i
            random_pick = random.uniform(0, 1) <= greedy_factor
            if random_pick:
                action = random.choice(action_list)
            else:
                output = model(get_state_input(state))
                action = torch.argmax(output).numpy().item()
            reward, next_frame, game_over = game.take_action(action)
            next_state = np.append(state[1:, :, :], np.expand_dims(next_frame, axis=0), axis=0)
            memory_buffer.append([state, action, reward, next_state, game_over])
            if len(memory_buffer) > memory_buffer_capacity:
                memory_buffer.pop(0)
            
            # print(f"greedy_factor: {greedy_factor}, random_pick: {random_pick}, action: {action}, game_over: {game_over}")
            
            # train model
            if total_steps % update_per_timesteps == 0:
                batch = random.sample(memory_buffer, min(len(memory_buffer), batch_size))
                action_batch = [e[1] for e in batch]
                x_batch = torch.stack([get_state_input(e[0]) for e in batch]).squeeze(1)
                y_batch = torch.tensor([
                    e[2] if e[4] \
                    else e[2] + discount_factor * torch.max(model(get_state_input(e[3]))).detach().numpy() \
                    for e in batch
                ]).float()

                # Compute prediction and loss
                pred = model(x_batch)[torch.arange(len(action_batch)), action_batch]
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
        
        # save model
        if (i + 1) % save_model_per_episodes == 0:
            print(f"save model on episode: {i}")
            torch.save(model.state_dict(), model_weights_path)
    
    game.close()

def test():
    model = Model()
    game = Game()

    model.load_state_dict(torch.load(model_weights_path))

    game.open()
    game.start()

    restarted = True

    while(True):
        frame = game.get_frame()
        game.display(frame)

        # take most recent 4 frames as state
        if restarted:
            state = np.stack((frame, frame, frame, frame))
            restarted = False
        else:
            state = np.append(state[1:, :, :], np.expand_dims(frame, axis=0), axis=0)

        output = model(get_state_input(state))
        action = torch.argmax(output).numpy()
        _, _, game_over = game.take_action(action)
        print(f"output: {output}, action: {action}")
        if game_over:
            game.restart()
            restarted = True