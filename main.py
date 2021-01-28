import time
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
from Connect4 import *

from keras import Model
from keras.layers import Dense, Conv3D, MaxPooling3D, Input, Softmax


##
game = Game()
game.add_tokens(11, 12, 12, 11)
game.display_board()
game.clear_board()

p1 = Player(game)
p2 = Player(game, 'Dean')

##
for k in range(5):
    game.clear_board()
    for i in range(10):
        p1.play()
        p2.play()
    game.display_board()
    game.check_connect4()

##

pool_b = []
pool_arr = []

for k in range(10000):
    pool_arr.append(npr.randint(2, size=(4, 4, 4), dtype='bool'))
    bits = npr.randint(2, size=100)
    a = 0
    deux = 1
    for bit in bits:
        a += deux * int(bit)
        deux *= 2
    pool_b.append(a)


## Temps bits

tab = np.zeros(10000, dtype='bool')
deltas = np.array([1, 4, 5, 6, 19, 20, 21, 24, 25, 26, 29, 30, 31])
ddeltas = 2*deltas
t0 = time.time()
for i, b in enumerate(pool_b):
    bb = b & (b >> deltas)
    tab[i] |= (bb & (bb >> ddeltas)).any()
print(time.time()-t0)

## Temps np array
tab = np.zeros(10000, dtype='bool')
t0 = time.time()
for i, b in enumerate(pool_arr):
    # Les grandes diagonales
    # tab[i] |= b[0, 0, 0] == b[1, 1, 1] == b[2, 2, 2] == b[3, 3, 3]
    # tab[i] |= b[3, 0, 0] == b[2, 1, 1] == b[1, 2, 2] == b[0, 3, 3]
    # tab[i] |= b[0, 3, 0] == b[1, 2, 1] == b[2, 1, 2] == b[3, 0, 3]
    # tab[i] |= b[3, 3, 0] == b[2, 2, 1] == b[1, 1, 2] == b[0, 0, 3]

    # tab[i] |= b[0, 0, 0] * b[1, 1, 1] * b[2, 2, 2] * b[3, 3, 3]
    # tab[i] |= b[3, 0, 0] * b[2, 1, 1] * b[1, 2, 2] * b[0, 3, 3]
    # tab[i] |= b[0, 3, 0] * b[1, 2, 1] * b[2, 1, 2] * b[3, 0, 3]
    # tab[i] |= b[3, 3, 0] * b[2, 2, 1] * b[1, 1, 2] * b[0, 0, 3]

    # # Les trucs droits
    #
    tab[i] |= b.all(axis=0).any()
    tab[i] |= b.all(axis=1).any()
    tab[i] |= b.all(axis=2).any()

print(time.time()-t0)
##


##

input_board = Input(shape=(4, 4, 4, 1))

x = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(input_board)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Softmax(16)(x)

autoencoder = Model(input_board, x)
