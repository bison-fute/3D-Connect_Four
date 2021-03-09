from MCTS import *
from CNN import *

# hyper-parameters
numIters = 1
numEps = 1
threshold = 0.6
numMCTSSims = 1


def policyIterSP(game):
    # nnet = initNNet()  # initialise random neural network
    nnet = 1
    examples = []
    for i in range(numIters):
        for e in range(numEps):
            examples += executeEpisode(game, nnet)  # collect examples from this game
        new_nnet = trainNNet(examples)
        frac_win = compare_nets(new_nnet, nnet)  # compare new net with previous net
        if frac_win > threshold:
            nnet = new_nnet  # replace with new net
    return nnet


def executeEpisode(game): #executeEpisode(game, nnet):
    examples = []
    s = game
    mcts = MCTS()  # initialise search tree

    while True:
        for _ in range(numMCTSSims):
            mcts.search(s, game) #mcts.search(s, game, nnet)
            print(mcts.visited)
        examples.append([s, mcts.pi(s), None])  # rewards can not be determined yet
        a = random.choice(len(mcts.pi(s)), p=mcts.pi(s))  # sample action from improved policy
        s = game.nextState(s, a)
        if game.gameEnded(s):
            examples = assignRewards(examples, game.gameReward(s))
            return examples

game = Game()
executeEpisode(game)

def compare_nets():
    return 0
