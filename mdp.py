from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import numpy as np
import random

class MDP():
    def __init__(self):
        """
        Transition : Dictionnaire d'actions, chaque action a sa propre matrice de transition
        """
        # List of states
        self.states = []

        # List of actions
        self.actions = []

        # Transition matrix
        self.transition = {}

    def summary(self):
        print("Markovian Decision Process Summary")
        print("----------------------------------")
        print("States : " + str(self.states))
        print("Actions : " + str(self.states))



    def print(self):
        pass

    def simulate(self, max_steps = 100):
        auto = input("Simulation automatique ? Y/N \n")
        print("\nStarting Simulation\n--------------------")
        print("Max transitions : " + str(max_steps)+ "\n--------------------")
        print("Starting states : " + str(self.states[0]))

        state = self.states[0]
        t = 0

        for step in range(max_steps):

            index = self.states.index(state)
            possible_actions = []
            for act in self.transition.keys():
                if np.sum(self.transition[act], axis=1)[index] != 0:
                    possible_actions.append(act)

            targets = []
            weights = []

            if None in possible_actions:
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]

                if len(targets) == 1 and targets[0] == state:
                    break
                if len(targets) == 0:
                    break

                choice = random.choices(targets, weights = weights)
                print("Transition from " + state + " to " + choice[0])
                state = choice[0]
                

            else:
                if auto == "N" :
                    act = input("Choisissez une action parmi les suivantes : " + str(possible_actions) + "\n")
                else:
                    act = random.choice(possible_actions)

                targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                weights = [self.transition[act][index][i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]

                if len(targets) == 1 and targets[0] == state:
                    break
                if len(targets) == 0:
                    break

                choice = random.choices(targets, weights = weights)
                print("Transition from " + state + " to " + choice[0])
                state = choice[0]

        
class gramPrintListener(gramListener):

    def __init__(self):
        self.mdp = MDP()
        
    def enterDefstates(self, ctx):
        states = [str(x) for x in ctx.ID()]
        self.mdp.states = states
        print("States: %s" % str(states))

    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.mdp.actions = actions
        print("Actions: %s" % str(actions))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))

        if act not in self.mdp.actions:
            self.mdp.actions.append(act)

        if act not in self.mdp.transition.keys():
            self.mdp.transition[act] = np.zeros((len(self.mdp.states), len(self.mdp.states)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[act][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

        if None not in self.mdp.transition.keys():
            self.mdp.transition[None] = np.zeros((len(self.mdp.states), len(self.mdp.states)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[None][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight

    @property
    def getMDP(self):
        return self.mdp

            

def main():
    lexer = gramLexer(FileStream("ex.mdp"))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)

    mdp = printer.getMDP
    print(mdp.transition[None])
    print(np.sum(mdp.transition[None], axis = 1))
    mdp.simulate()


if __name__ == '__main__':
    main()
