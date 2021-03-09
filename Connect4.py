import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import torch


def seize_to_pos(n):
    """
    Transforme un nombre entre 0 et 15 en une position valable
    pour Game.add_tokens
    :param n:
    :return:
    """
    return 10*(n//4) + n % 4

def pos_to_seize(pos):
    """
    Fonction réciproque de la précédente
    :param pos:
    :return:
    """
    return 4*(pos//10) + (pos % 10)

def xyz_to_bin_pos(x, y, z):
    """
    Transforme des coordonnées 3D en 1D
    :param x:
    :param y:
    :param z:
    :return:
    """
    return x + 4 * y + 16 * z


def binpos_to_xyz(_bin_pos):
    """
    Fonction réciproque de la précédente
    :param _bin_pos:
    :return:
    """
    x = _bin_pos % 4
    y = ((_bin_pos - x) % 16) / 4
    z = (_bin_pos - y * 4 - x) / 16
    return int(x), int(y), int(z)


class Game:
    """
        Classe réprésentant le plateau de jeu et les différentes actions possibles
        pour les joueurs
    """
    def __init__(self):
        # Réprésentation naturelle du plateau de jeu -> 0: case vide, 1: jeton joueur 1, -1: jeton joueur 2
        self.board = np.zeros((4, 4, 4))

        # États utilisés par les réseaux de neuronnes
        self.board_j1 = torch.zeros(64)
        self.board_j2 = torch.zeros(64)

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
            bin_pos = (i//4) * 10 + (i % 4)
            self.free_positions.append(bin_pos)
        # free_positions code les positions libre (x, y) pour ajouter un jetons sur le plateau
        # (une position devient non disponible une fois que la colonne correspondante est remplie)

        # dictionnaire associant un jeton à ses trios d'amis, avec lesquels il peut faire un puissance 4
        self.dict_friends = {}

        # La fonction pour compléter le dictionnaire d'amis
        # Ne pas regarder si vous ne voulez pas pleurer du sang
        def complete_friends(_bin_pos):
            friends = []
            x, y, z = binpos_to_xyz(_bin_pos)
            tempx, tempy, tempz = [], [], []

            # Les lignes, colonnes, rangés
            for k in range(4):
                if k != x:
                    tempx.append(xyz_to_bin_pos(k, y, z))
                if k != y:
                    tempy.append(xyz_to_bin_pos(x, k, z))
                if k != z:
                    tempz.append(xyz_to_bin_pos(x, y, k))
            friends.append(tempx)
            friends.append(tempy)
            friends.append(tempz)

            # Les 6 configurations de diagonales 2D
            if x == y:
                temp = []
                for k in range(4):
                    if k != x:
                        temp.append(xyz_to_bin_pos(k, k, z))
                friends.append(temp)

            if x == 3-y:
                temp = []
                for k in range(4):
                    if k != x:
                        temp.append(xyz_to_bin_pos(k, 3-k, z))
                friends.append(temp)

            if x == z:
                temp = []
                for k in range(4):
                    if k != x:
                        temp.append(xyz_to_bin_pos(k, y, k))
                friends.append(temp)

            if x == 3-z:
                temp = []
                for k in range(4):
                    if k != x:
                        temp.append(xyz_to_bin_pos(k, y, 3-k))
                friends.append(temp)

            if y == z:
                temp = []
                for k in range(4):
                    if k != y:
                        temp.append(xyz_to_bin_pos(x, k, k))
                friends.append(temp)

            if y == 3-z:
                temp = []
                for k in range(4):
                    if k != y:
                        temp.append(xyz_to_bin_pos(x, k, 3-k))
                friends.append(temp)

            # les 4 grandes diagonales 3D
            d_pp = [0, 21, 42, 63]
            d_pm = [3, 22, 41, 60]
            d_mp = [12, 25, 38, 51]
            d_mm = [15, 26, 37, 48]
            if _bin_pos in d_pp:
                d_pp.remove(_bin_pos)
                friends.append(d_pp)
            if _bin_pos in d_pm:
                d_pm.remove(_bin_pos)
                friends.append(d_pm)
            if _bin_pos in d_mp:
                d_mp.remove(_bin_pos)
                friends.append(d_mp)
            if _bin_pos in d_mm:
                d_mm.remove(_bin_pos)
                friends.append(d_mm)

            all_friends = []
            for trio in friends:
                all_friends += trio
            self.dict_friends[_bin_pos] = all_friends
            return None

        # la complétion
        for bin_pos in range(64):
            complete_friends(bin_pos)

        # liste contenant la valeur de la position du joueur 1, depuis le début de
        # la partie
        self.position_value = [0]


    def display_board(self, title=None):
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
        if title is not None:
            plt.title(title)
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
        _z = 0
        while self.board[_x, _y, _z] != 0:
            _z += 1
            if _z == 4:
                break
        if _z < 4:          # Ajout du jeton
            self.board[_x, _y, _z] = _player
            if _player == 1:
                self.binboard_j1 += 2**(_x + 5*_y + 25*_z)
                self.binboard_j1 = int(self.binboard_j1)
                self.board_j1[_x + 4 * _y + 16 * _z] = 1
            else:
                self.binboard_j2 += 2**(_x + 5*_y + 25*_z)
                self.binboard_j2 = int(self.binboard_j2)
                self.board_j2[_x + 4 * _y + 16 * _z] = 1
            self.ntok += 1
            self.update_pos_val(_x + 4 * _y + 16 * _z, _player)
            if _z == 3:
                self.free_positions.remove(pos)
        else:
            raise ValueError('Colonne remplie')
        return None

    def remove_token(self, pos):
        """
            Cf self.add_token
        """
        _x, _y = pos // 10, pos % 10
        if self.ntok % 2 == 0:
            _player = -1
        else:
            _player = 1
        free = False
        _z = 0
        while self.board[_x, _y, _z] != 0:
            _z += 1
            if _z == 4:
                break
        _z -= 1
        if _z == -1:
            raise ValueError('Colonne Vide')
        else:  # Ajout du jeton
            self.board[_x, _y, _z] = 0
            if _player == 1:
                self.binboard_j1 -= 2 ** (_x + 5 * _y + 25 * _z)
                self.binboard_j1 = int(self.binboard_j1)
                self.board_j1[_x + 4*_y + 16*_z] = 0
            else:
                self.binboard_j2 -= 2 ** (_x + 5 * _y + 25 * _z)
                self.binboard_j2 = int(self.binboard_j2)
                self.board_j2[_x + 4 * _y + 16 * _z] = 0
            self.ntok -= 1
            self.position_value.pop()
            if _z == 3:
                self.free_positions.append(pos)
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

    def get_turn(self):
        return self.ntok % 2

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
        self.board_j1 = torch.zeros(64)
        self.board_j2 = torch.zeros(64)
        self.binboard_j1 = 0
        self.binboard_j2 = 0
        self.ntok = 0
        self.free_positions = []
        for i in range(16):
            pos = (i//4) * 10 + (i % 4)
            self.free_positions.append(pos)

    def reset(self):
        self.clear_board()
        return self.get_state()

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

    def get_state(self):
        if self.ntok % 2 == 0:
            return torch.cat([self.board_j1, self.board_j2])
        else:
            return torch.cat([self.board_j2, self.board_j1])

    def check_winning_move(self):
        raise ValueError('To implement')

    def update_pos_val(self, _bin_pos, player):
        """
        Met à jour la position value
        :param _bin_pos: the position of the new token added
        :param player: 1 if player 1, -1 if player 2
        :return:
        """
        friends = self.dict_friends[_bin_pos]
        last_val = self.position_value[-1]

        s_j1 = self.board_j1[friends].reshape(-1, 3).sum(axis=1)
        print("s_j1\n", s_j1)
        s_j2 = self.board_j2[friends].reshape(-1, 3).sum(axis=1)
        print("s_j2\n", s_j2)
        if player == 1:
            new_val = last_val + ((s_j2 == 0).sum() + np.dot(s_j2, (s_j1 == 0))).item()
        else:
            new_val = last_val - ((s_j1 == 0).sum() + np.dot(s_j1, (s_j2 == 0))).item()
        self.position_value.append(int(new_val))

    def get_position_value(self):
        return self.position_value[-1]


g = Game()
##
print('Ajout token')
g.add_tokens(33)
print(g.position_value)
print('Ajout token')
g.add_tokens(30)
print(g.position_value)
print('Ajout token')
g.add_tokens(31)
print(g.position_value)

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
