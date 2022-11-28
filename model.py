import torch
import random
import time
import numpy as np
from torch import nn
from pathlib import Path
from dino import Game, num_actions, action_list
from utils import get_logger, get_top_models, save_state_as_image

file_dir = Path(__file__).parent

model_weights_file = "model_weights.pth"
model_weights_dir = file_dir / "trained_model"
model_weights_path = model_weights_dir / model_weights_file

# train params
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

# validate params
num_top_models = 100
num_episodes_validate = int(2e1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.neural_network = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 2, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(672, 256),
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

# get state input of shape (1, 1, frame_shape[0], frame_shape[1]) for model
def get_state_input(state):
    state_input = torch.from_numpy(state).type(torch.float32).unsqueeze(0).unsqueeze(0)
    return state_input

def train():
    logger = get_logger("train")
    model = Model()
    game = Game()

    # load pretrained model
    if (model_weights_path.exists()):
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.apply(init_weights)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    total_steps = 0
    memory_buffer = []
    episode_steps = []

    game.open()
    game.start()

    for i in range(num_episodes):
        for t in range(maximum_episode_length):
            # frame_start_time = time.time()
            total_steps += 1
            frame = game.get_frame()
            game.display(frame)
            
            # take next action
            greedy_factor = init_greedy_factor - \
                (init_greedy_factor - final_greedy_factor) / num_episodes * i
            random_pick = random.uniform(0, 1) <= greedy_factor
            if random_pick:
                action = random.choice(action_list)
            else:
                output = model(get_state_input(frame))
                action = torch.argmax(output).numpy().item()
            reward, next_frame, game_over = game.take_action(action)
            memory_buffer.append([frame, action, reward, next_frame, game_over])
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
                logger.info(f"episode: {i}, step: {t}, loss: {loss}")
            
            # save_state_as_image(i, t, frame, action, next_frame, game_over)
            # frame_end_time = time.time()
            # frame_rate = 1 / (frame_end_time - frame_start_time)
            # print(f"frame_rate: {frame_rate}")

            if game_over or t == maximum_episode_length - 1:
                logger.info(f"episode: {i}, episode_steps: {t}")
                episode_steps.append([i, t])
                game.restart()
                break
        
        # save model
        if (i + 1) % save_model_per_episodes == 0:
            average_steps = np.average(np.array(episode_steps)[-save_model_per_episodes:, 1].astype(np.float))
            logger.info(f"save model on episode: {i}, average_steps: {average_steps} (in recent {save_model_per_episodes} games)")
            new_model_weights_path = model_weights_dir / f"model_weights_{i}.pth"
            torch.save(model.state_dict(), new_model_weights_path)
    
    game.close()

def validate():
    logger = get_logger("validate")
    model = Model()
    game = Game()

    model_episode_steps = []
    model_validate_res = []
    top_models = get_top_models(num_top_models)

    game.open()
    game.start()
    
    for m in top_models:
        model.load_state_dict(torch.load(model_weights_dir / m))

        for i in range(num_episodes_validate):
            game_over = False
            steps = 0
            while(not game_over):
                frame = game.get_frame()
                game.display(frame)
                output = model(get_state_input(frame))
                action = torch.argmax(output).numpy()
                _, _, game_over = game.take_action(action)
                if game_over:
                    model_episode_steps.append([m, i, steps])
                    logger.info(f"model: {m}, episode: {i}, episode_steps: {steps}")
                steps += 1
            game.restart()
        
        model_steps = np.array(model_episode_steps)[-num_episodes_validate:, 2].astype(np.float)
        average_steps = np.average(model_steps)
        std = np.std(model_steps)
        
        model_validate_res.append([m, average_steps, std])
        logger.info(f"model: {m}, average_steps: {average_steps}, std: {std}")
    
    game.close()

    sorted_model_validate_res = sorted(model_validate_res, key=lambda res: (-res[1], res[2]))
    logger.info(f"model ranking: {sorted_model_validate_res}")

def test():
    model = Model()
    game = Game()

    model.load_state_dict(torch.load(model_weights_path))

    game.open()
    game.start()

    while(True):
        frame = game.get_frame()
        output = model(get_state_input(frame))
        action = torch.argmax(output).numpy()
        _, _, game_over = game.take_action(action)
        if game_over:
            game.restart()