import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, MaxPooling2D, UpSampling2D
from keras import Model



class Game:
    """
        Classe réprésentant le plateau de jeu et les différentes actions possibles
        pour les joueurs
    """
    def __init__(self):
        # Réprésentation naturelle du plateau de jeu -> 0: case vide, 1: jeton joueur 1, -1: jeton joueur 2
        self.board = np.zeros((4, 4, 4))

        # Réprésentation binaire des positions des jetons du premier joueur
        # (ça permet de vérifier rapidement s'il y a un puissance 4)
        self.binboard_j1 = 0
        # De même pour le deuxième joueur
        self.binboard_j2 = 0

        # Nombre de jetons sur le plateau (2 joueurs confondus)
        self.ntok = 0

        self.free_positions = []
        # On définit les indices de décalage pour check s'il y a un puissance 4
        self.deltas = np.array([1, 4, 5, 6, 19, 20, 21, 24, 25, 26, 29, 30, 31], dtype=object)
        self.deltas2 = 2*self.deltas                          # Pour économiser des calculs
        for i in range(16):
            pos = (i//4) * 10 + (i % 4)
            self.free_positions.append(pos)
        # free_positions code les positions libre (x, y) pour ajouter un jetons sur le plateau
        # (une position devient non disponible une fois que la colonne correspondante est remplie)


    def display_board(self):
        """
            Fonction utilitaire pour afficher le plateau de jeu
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axes.set_xlim3d(left=0.2, right=2.8)
        ax.axes.set_ylim3d(bottom=0.2, top=2.8)
        ax.axes.set_zlim3d(bottom=0.2, top=2.8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_zticks([0, 1, 2, 3])
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if self.board[i, j, k] == 1:
                        ax.scatter(i, j, k, c='yellow', s=100)
                    if self.board[i, j, k] == -1:
                        ax.scatter(i, j, k, c='red', s=100)
        plt.show()
        return None

    def __add_token(self, pos):
        """
            Cf self.add_token
        """
        _x, _y = pos // 10, pos % 10
        if self.ntok % 2 == 0:
            _player = 1
        else:
            _player = -1
        free = False
        _z = 0
        while self.board[_x, _y, _z] != 0:
            _z += 1
            if _z == 4:
                break
        if _z < 4:          # Ajout du jeton
            self.board[_x, _y, _z] = _player
            if _player == 1:
                self.binboard_j1 += 2**(_x + 5*_y + 25*_z)
            else:
                self.binboard_j2 += 2**(_x + 5*_y + 25*_z)
            self.ntok += 1
            if _z == 3:
                self.free_positions.remove(pos)
        else:
            raise ValueError('Colonne remplie')
        return None


    def add_tokens(self, *args):
        """
            Fonction pour ajouter un (ou des) jeton(s) au plateau
            (en alternant entre les joueurs):

            > g = Game()
            # Crée un plateau vide
            > g.add_token(11)
            # Ajoute un jeton pour le joueur 1 dans la colonne (1, 1)
            > add_token(12, 21, 23)
            # Ajoute un jeton pour le joueur 2 en (1, 2)
            # Puis un autre jeton pour le joueur 1 en (2, 1)
            # Puis un autre jeton pour le joueur 2 en (2, 3)
        """
        for pos in args:
            self.__add_token(pos)
        return None

    def get_ntokens(self):
        return self.ntok

    def end_game(self):
        """
        Détermine si le plateau est rempli
        :return: boolean
        """
        return self.ntok == 64

    def clear_board(self):
        """
        Vide le plateau
        :return: None
        """
        self.board = np.zeros((4, 4, 4))
        self.binboard_j1 = 0
        self.binboard_j2 = 0
        self.ntok = 0
        self.free_positions = []
        for i in range(16):
            pos = (i//4) * 10 + (i % 4)
            self.free_positions.append(pos)

    def check_connect4(self, show=True):
        """
        Cheack s'il y a puissance 4
        :param show: Paramètre d'affichage
        :return: boolean
        """
        bb1 = self.binboard_j1 & (self.binboard_j1 >> self.deltas)
        bb2 = self.binboard_j2 & (self.binboard_j2 >> self.deltas)
        if show:
            print('Joueur 1 :', bool((bb1 & (bb1 >> self.deltas2)).any()))
            print('Joueur 2 :', bool((bb2 & (bb2 >> self.deltas2)).any()))
        else:
            return (bb1 & (bb1 >> self.deltas2)).any() | (bb2 & (bb2 >> self.deltas2)).any()


## Random player
class Player:
    """
        Dummy player 1
    """
    def __init__(self, _game, name=None):
        self.game = _game
        if name:
            self.name = name
        else:
            self.name = 'Todd'
        print(self.name, 'is ready to play')

    def play(self):                 # Random
        if self.game.end_game():
            raise ValueError('Partie finie')
        k = npr.randint(len(self.game.free_positions))
        self.game.add_tokens(self.game.free_positions[k])


##
"""
Idées : faire des convolutions sur les 54 structures 1D et / ou
sur les 48 structures 2D
(En tout cas les convolutions normales sembles 
moins pertienentes... sauf pour le centre)
"""
