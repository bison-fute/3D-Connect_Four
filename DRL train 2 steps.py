from DRL import *
from Connect4 import *

##
import torch
import torch.nn as nn
import numpy.random as npr
from random import sample
import torch.nn.functional as F


##
n_epochs = 10000

replay_memory_size = 1000
tau_ini = 1
tau_fin = 1
batch_size = 64
gamma = 0.9
##
agent = DQNAgent('Chamby')
# todd = DQNAgent('Todd')         # Joueur 1
# alice = DQNAgent('Alice')       # Joueur 2
# agents = [todd, alice]
current_game = Game()
# past_state = None
# past_action = None
temp = [None, None]                         # (past_state, past_action)
replay_memory = []
##

def get_action(state, game, _tau=1.):
    # Plus tard : jouer le coup gagnat si disponible
    probas = F.softmax(agent(state) * _tau, dim=0).detach().numpy()
    return npr.choice(len(probas), p=probas)

def get_action_eps(state, game, _eps=0):
    # Plus tard : jouer le coup gagnat si disponible
    if npr.rand() < _eps:
        ind = npr.randint(len(game.free_positions))
        return pos_to_seize(game.free_positions(ind))
    else:
        return agent(state).detach().numpy().argmax()

# def get_next_state_and_reward(game, action):
#     """
#     Si état final, renvoie un plateau vide
#     Reward de +1 si victoire, -1 si défaite au prochain coup, -10 si coup invalide
#     :param game:
#     :param action:
#     :return:
#     """
#     reward = 0
#     pos = seize_to_pos(action)
#     final = False
#     if pos not in game.free_positions:     # L'agent a joué un coup illégal
#         reward = -100.
#         # final = True
#         # game.clear_board()
#     else:
#         game.add_tokens(pos)
#
#         if game.check_connect4(show=False):          # On vient de gagner
#
#             reward = 1.
#             final = True
#             game.clear_board()
#         elif game.end_game():
#             reward = -0.1
#             final = True
#             game.clear_board()
#
#     new_state = game.get_state()
#     return new_state, reward, final

def add_replay_move(game, _tau=None, _eps=None):
    # Ajout d'un nouveau coup à la base de données
    state = game.get_state()
    if _tau is not None:
        action = get_action(state, _tau=_tau)
    else :
        action = get_action_eps(state, game, _eps)
    pos = seize_to_pos(action)

    try_count = 0
    while pos not in game.free_positions:       # L'agent a joué un coup illégal
        replay_memory.append((state, action, state, -100., True))
        try_count += 1
        if try_count < 10:
            action = get_action(state, _tau=_tau)
        else:
            ind = npr.randint(len(game.free_positions))
            pos = game.free_positions[ind]
            action = pos_to_seize(pos)

    game.add_tokens(pos)

    if game.check_connect4(show=False):  # On vient de gagner
        replay_memory.append((temp[0], temp[1], temp[0], -1., True))
        replay_memory.append((state, action, state, 1., True))
        game.clear_board()
        temp[0] = None
        temp[1] = None

    if game.end_game():
        game.clear_board()
        replay_memory.append((temp[0], temp[1], temp[0], -0.1, True))
        replay_memory.append((state, action, state, -0.1, True))

    new_state = game.get_state()
    if temp[0] is not None:
        replay_memory.append((temp[0], temp[1], new_state, 0., False))
    temp[0] = new_state
    temp[1] = action

def compute_q(agent_output, _actions):
    return agent_output[torch.arange(agent_output.shape[0]), _actions]

def compute_targets_q(_next_states, _rewards, _finals, _gamma):
    return _rewards + _gamma * _finals * torch.max(agent(_next_states), dim=1)[0]


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

    agent.optimizer.zero_grad()
    loss = F.mse_loss(q_values, target_q_values)
    loss.backward()

    loss_mean += loss.item() / batch_size
    inv_play_mean += finals.sum().float().item() / batch_size
    loss_history.append(loss_mean / batch_size / (epoch+1))
    inv_play_history.append(inv_play_mean / (epoch+1))

    print("\rEpoch %d/%d, loss = %.4f, invalid rate = %.4f"
          % (epoch,
             n_epochs,
             loss_mean/(epoch+1),
             inv_play_mean / (epoch+1)), end='')

print()
##
plt.plot(loss_history)
plt.title('loss_history')
plt.show()

plt.plot(inv_play_history)
plt.title('invalid rate history')
plt.show()
##

# À toi de jouer

g = Game()


##
agent(g.get_state()).view(4, 4)

##
print(agent(g.get_state()).view(4, 4))
n = torch.argmax(agent(g.get_state())).item()
print(10 * (n//4) + (n % 4))
##

