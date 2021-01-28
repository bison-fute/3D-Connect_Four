import time
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
from Connect4 import *

from keras import Model
from keras.layers import Dense, Conv3D, MaxPooling3D, Input, Softmax


##

# Exemple de début de jeu, avec affichage du plateau
game = Game()
game.add_tokens(11, 12, 12, 11)
game.display_board()
game.clear_board()

p1 = Player(game)
p2 = Player(game, 'Dean')

##

# Quelques débuts de parties s'arrêtant au tour 10, avec vérification de puissance 4
for k in range(5):
    game.clear_board()
    for i in range(10):
        p1.play()
        p2.play()
    game.display_board()
    game.check_connect4()

##

# Début de model, rien d'important
input_board = Input(shape=(4, 4, 4, 1))

x = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(input_board)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Softmax(16)(x)

autoencoder = Model(input_board, x)
