import numpy as np
from numpy import random
import torch
import torch.nn as nn
import re
import copy

# hyper-parameters
exploration_factor = 1

positions = {0: 0, 1: 1, 2: 2, 3: 3,
             10: 4, 11: 5, 12: 6, 13: 7,
             20: 8, 21: 9, 22: 10, 23: 11,
             30: 12, 31: 13, 32: 14, 33: 15,
             }


class MCTS:

    def __init__(self):
        # start dictionary for MCTS exploration
        # keys: strings corresponding to the board
        # values: [Q values for the state, number of visits]
        # first key is empty board
        self.x_0 = np.zeros(64)
        self.key_0 = np.array2string(self.x_0)
        self.visited = {self.key_0: [np.zeros(16), np.zeros(16)]}

    def search(self, game): #(self, game, nnet):
        # we return -1 here because it's the turn following the win
        if game.check_connect4():
            return -1
        state = np.array2string(game.board)

        if state not in self.visited:
            self.visited.update({state: [np.zeros(16), np.zeros(16)]})
            # replace by nn.prediction !!
            # P[s], v = nnet.prediction(state)
            prediction, v = np.random.rand(16) * 2 - 1, \
                            np.random.randint(0, 2, 1)[0] * 2 - 1
            return -v

        # replace by nn prediction !!
        prediction, v = np.random.rand(16) * 2 - 1, \
                        np.random.randint(0, 2, 1)[0] * 2 - 1

        max_u, best_a = -float("inf"), -1
        N = np.sum(self.visited.get(state)[1])
        for a in game.free_positions():
            # u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
            Q = self.visited.get(state)[0][positions.get(a)]
            P = prediction[positions.get(a)]
            N_a = self.visited.get(state)[1][positions.get(a)]
            u = (Q + exploration_factor * P * np.sqrt(N)) / (N_a + 1)
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        game.add_tokens(a)
        v = self.search(game)  #game, nnet

        Q = self.visited.get(state)[0][positions.get(a)]
        N_a = self.visited.get(state)[1][positions.get(a)]
        self.visited.get(state)[0][positions.get(a)] = (N_a * Q + v) / (N_a + 1)
        self.visited.get(state)[1][a] += 1
        return -v


# auxiliary function to retrieve array from string key in dictionary
def string2array(string):
    string = re.sub(r'\W+', '', string)
    array = np.array(list(string)).astype(int)
    return array
