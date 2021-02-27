from DRL import *
import torch
import torch.nn as nn
import numpy.random as npr
from random import sample
import torch.nn.functional as F
from Connect4 import *

##
n_epochs = 100000
replay_memory = []
replay_memory_size = 100
tau_ini = 1
tau_fin = 100
batch_size = 64
gamma = 0.9

##
agent = DQNAgent(' ')
todd = DQNAgent('Todd')         # Joueur 1
alice = DQNAgent('Alice')
agents = [todd, alice]
current_game = Game()

##

def get_action(_state, _tau=1.):
    # Plus tard : jouer le coup gagnat si disponible
    probas = F.softmax(agent(_state) * _tau, dim=0).detach().numpy()
    return npr.choice(len(probas), p=probas)

def get_next_state_and_reward(game, action):
    """
    Si état final, renvoie un plateau vide
    Reward de +1 si victoire, -1 si défaite au prochain coup, -10 si coup invalide
    :param game:
    :param action:
    :return:
    """
    reward = 0
    pos = seize_to_pos(action)
    final = False
    if pos not in game.free_positions:     # L'agent a joué un coup illégal
        reward = -100.
        final = True
        game.clear_board()
    else:
        game.add_tokens(pos)

        if game.check_connect4(show=False):          # On vient de gagner

            reward = 1.
            final = True
            game.clear_board()
        elif game.end_game():
            reward = -0.1
            final = True
            game.clear_board()

    new_state = game.get_state()
    return new_state, reward, final

def add_replay_move(game, _tau):
    # Ajout d'un nouveau coup à la base de données
    _state = game.get_state()
    action = get_action(_state, _tau=_tau)
    next_state, reward, final = get_next_state_and_reward(game, action)
    replay_memory.append((_state, action, next_state, reward, final))

def compute_q(agent_output, _actions):
    return agent_output[torch.arange(agent_output.shape[0]), _actions]


def compute_targets_q(_next_states, _rewards, _finals, _gamma):
    return _rewards - _gamma * (1-1*_finals) * torch.max(agent(_next_states), dim=1)[0]


for _ in range(replay_memory_size):
    add_replay_move(current_game, tau_ini)


loss_mean = 0
inv_play_mean = 0
loss_history = []
inv_play_history = []
for epoch in range(n_epochs):
    tau = tau_ini + epoch * (tau_fin - tau_ini) / n_epochs

    # Ajout d'un nouveau coup
    add_replay_move(current_game, tau)
    replay_memory.pop(0)

    # Entrainement
    batch = sample(replay_memory, batch_size)
    states = torch.stack(tuple(d[0] for d in batch))
    actions = torch.tensor([d[1] for d in batch])
    next_states = torch.stack(tuple(d[2] for d in batch))
    rewards = torch.tensor([d[3] for d in batch])
    finals = torch.tensor([d[4] for d in batch])

    q_values = compute_q(agent(states), actions)
    target_q_values = compute_targets_q(next_states, rewards, finals, gamma)
    target_q_values.detach()

    # print(q_values.view(-1, 1))
    # print(target_q_values.view(-1, 1))
    # 4/0
    agent.optimizer.zero_grad()
    loss = F.mse_loss(q_values, target_q_values)
    loss.backward()

    loss_mean += loss.item()
    inv_play_mean += (rewards == -100.).sum().float() / batch_size

    loss_history.append(loss_mean/(epoch+1))
    inv_play_history.append(inv_play_mean / (epoch+1))
    print("\rEpoch %d/%d, loss = %.4f, invalid rate = %.4f"
          % (epoch,
             n_epochs,
             loss_mean/(epoch+1),
             inv_play_mean / (epoch+1)), end='')
print()

##

# À toi de jouer

g = Game()


##
agent(g.get_state()).view(4, 4)

##
print((agent(g.get_state()).view(4, 4)*100).int())
n = torch.argmax(agent(g.get_state())).item()
print(10 * (n//4) + (n % 4))
##

