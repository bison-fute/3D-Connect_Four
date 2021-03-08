import numpy as np
from numpy import random
import torch
import torch.nn as nn
import re
import copy

# hyper-parameters
c = 1
numIters = 1

positions = {0: 0, 1: 1, 2: 2, 3: 3,
             10: 4, 11: 5, 12: 6, 13: 7,
             20: 8, 21: 9, 22: 10, 23: 11,
             30: 12, 31: 13, 32: 14, 33: 15,
             }

# start dictionary for MCTS exploration
# keys: strings corresponding to the board
# values: [Q values for the state, number of visits]

# first key is empty board
x_0 = np.zeros(64)
key_0 = np.array2string(x_0)
visited = {key_0: [np.zeros(16), np.zeros(16)]}


# function to retrieve array from string key in dictionary
def string2array(string):
    string = re.sub(r'\W+', '', string)
    array = np.array(list(string)).astype(int)
    return array


def search(game, nnet):
    # we return -1 here because it's the turn following the win
    if game.check_connect4(): return -1
    state = np.array2string(game.board)

    if state not in visited:
        visited.update({state: [np.zeros(16), np.zeros(16)]})
        # replace by nn.prediction !!
        # P[s], v = nnet.prediction(state)
        prediction, v = np.random.rand(16) * 2 - 1, \
                        np.random.randint(0, 2, 1)[0] * 2 - 1
        return -v

    # replace by nn prediction !!
    prediction, v = np.random.rand(16) * 2 - 1, \
                    np.random.randint(0, 2, 1)[0] * 2 - 1

    max_u, best_a = -float("inf"), -1
    N = np.sum(visited.get(state)[1])
    for a in game.free_positions():
        # u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
        Q = visited.get(state)[0][positions.get(a)]
        P = prediction[positions.get(a)]
        N_a = visited.get(state)[1][positions.get(a)]
        u = (Q + c * P * np.sqrt(N)) / (N_a + 1)
        if u > max_u:
            max_u = u
            best_a = a
    a = best_a

    game.add_tokens(a)
    v = search(game, nnet)

    Q = visited.get(state)[0][positions.get(a)]
    N_a = visited.get(state)[1][positions.get(a)]
    visited.get(state)[0][positions.get(a)] = (N_a * Q + v) / (N_a + 1)
    visited.get(state)[1][a] += 1
    return -v


def policyIterSP(game):

    # nnet = initNNet()  # initialise random neural network
    nnet = 1
    examples = []
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet)  # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = pit(new_nnet, nnet)  # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet  # replace with new net
    return nnet


def executeEpisode(game, nnet):
    examples = []
    s = game.startState()
    mcts = MCTS()  # initialise search tree

    while True:
        for _ in range(numMCTSSims):
            mcts.search(s, game, nnet)
        examples.append([s, mcts.pi(s), None])  # rewards can not be determined yet
        a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))  # sample action from improved policy
        s = game.nextState(s, a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s))
            return examples