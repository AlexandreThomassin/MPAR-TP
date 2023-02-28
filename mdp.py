from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import numpy as np
import random

import pydot
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

        # Possible actions for every state
        self.possible_actions = {}

        # Accessible states
        self.accessible = []

        self.graph = ""

        self.blue_state = ""

    def summary(self):
        print("Markovian Decision Process Summary")
        print("----------------------------------")
        print("States : " + str(self.states))
        print("Actions : " + str(self.actions))
        print(self.accessible)
        if sorted(self.accessible) == sorted(self.states):
            print("Tous les états sont accessibles")
        else:
            print(f"Les états suivants ne sont pas accessible : {list(set(self.states) - set(self.accessible))}")

    def MDP_test(self):
        
        # Test if state has transition with actions and without actions
        for state in self.states:
            index = self.states.index(state)
            if state not in self.possible_actions.keys():
                    print(f"WARNING : The state {state} has no transitions to other state.")
                    print("By default a transition on himself is added.\n")
                    self.transition[None][index][index] = 1
                    self.possible_actions[state] = [None]

            if None in self.possible_actions[state] and len(self.possible_actions[state]) != 1:
                        print(f"WARNING : The state {state} has transition both with actions and without actions !")
                        print("When simulate it will choose the path without actions.\n")

            
                    
        if set(self.accessible) != set(self.states):
            print(f"WARNING : Les états suivants ne sont pas accessible : {list(set(self.states) - set(self.accessible))}\n")

    def print(self):
        self.graph = pydot.Dot('Markov Chain Representation', graph_type='graph', bgcolor='white')
        states_graph = [pydot.Node(state, label = state) for state in self.states]
        for state in states_graph: self.graph.add_node(state)
        for source in self.states:
            index = self.states.index(source)
            possible_actions = self.possible_actions[source]
            
            if None in possible_actions:
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                for i in range(len(targets)):
                    self.graph.add_edge(pydot.Edge(source, targets[i], color='black', label = weights[i], arrowhead = 'normal', dir='forward'))
            else:
                for act in possible_actions:
                    self.graph.add_node(pydot.Node(f"{source}-{act}", shape = "point"))
                    self.graph.add_edge(pydot.Edge(source, f"{source}-{act}", color='black', label = act, arrowhead = 'normal', dir='forward'))
                    targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                    weights = [self.transition[act][index][i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                    for i in range(len(targets)):
                        self.graph.add_edge(pydot.Edge(f"{source}-{act}", targets[i], color='black', label = weights[i], arrowhead = 'normal', dir='forward'))
        png_str = self.graph.create_png(prog='dot')

        # treat the DOT output as an image file
        sio = io.BytesIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)

        # plot the image

        fig = plt.figure(figsize=(9,7))

        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

        imgplot = plt.imshow(img, aspect='equal')
        plt.axis("off")
        plt.title("Simulation \n", fontdict=font)
        plt.pause(0.01)
        plt.ion()
        plt.show()
        return

    def update(self, c_state):
        if self.blue_state in self.states:
            self.graph.add_node(pydot.Node(self.blue_state, label = self.blue_state, style = ""))
        if c_state in self.states:
            self.graph.add_node(pydot.Node(c_state, label = c_state, style = "filled", fillcolor = 'blue'))
            self.blue_state = c_state
        png_str = self.graph.create_png(prog='dot')

        # treat the DOT output as an image file
        sio = io.BytesIO()
        sio.write(png_str)
        sio.seek(0)
        img = mpimg.imread(sio)

        # plot the image

        font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

        plt.rcParams["figure.figsize"] = (9,7)

        imgplot = plt.imshow(img, aspect='equal')
        plt.axis("off")
        plt.title("Simulation \n", fontdict=font)
        plt.pause(0.01)
        plt.ion()
        plt.show()

    def simulate(self, max_steps = 100):
        auto = input("Simulation automatique ? Y/N \n")
        animate = input("Animation graphique? Y/N \n")
        print("\nStarting Simulation\n--------------------")
        print("Max transitions : " + str(max_steps)+ "\n--------------------")
        print("Starting states : " + str(self.states[0]))

        state = self.states[0]

        for step in range(max_steps):

            index = self.states.index(state)
            if animate == 'Y':
                self.update(state)

            targets = []
            weights = []

            if None in self.possible_actions[state]:
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]

                if len(targets) == 1 and targets[0] == state:
                    print("This state loop on himself, ending simulation.")
                    break

                choice = random.choices(targets, weights = weights)
                print("Transition from " + state + " to " + choice[0])
                state = choice[0]
                

            else:
                loop = True
                for act in self.possible_actions[state]:
                    targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                    if not (len(targets) == 1 and targets[0] == state):
                         loop = False
                       
                if loop:
                    print("This state loop on himself, ending simulation.")
                    break

                if auto == "N" :
                    act = input("Choisissez une action parmi les suivantes : " + str(self.possible_actions[state]) + "\n")
                    while act not in self.possible_actions[state]:
                        print("L'action ne fais pas parti des actions possible, veuillez ressayez")
                        act = input("Choisissez une action parmi les suivantes : " + str(self.possible_actions[state]) + "\n")
                else:
                    act = random.choice(self.possible_actions[state])

                targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                weights = [self.transition[act][index][i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]

                choice = random.choices(targets, weights = weights)
                print("Transition from " + state + " to " + choice[0])
                state = choice[0]
        if animate == 'Y':
            self.update(state)

        
class gramPrintListener(gramListener):

    def __init__(self):
        self.mdp = MDP()

    # Enter a parse tree produced by gramParser#program.
    def enterProgram(self, ctx:gramParser.ProgramContext):
        print("Begin Parsing")
        print("-------------\n")

    # Exit a parse tree produced by gramParser#program.
    def exitProgram(self, ctx:gramParser.ProgramContext):
    
        print("\n--------------")
        print("End of parsing")
        print("")
        print("Begin testing")
        print("--------------")
        self.mdp.MDP_test()
        print("--------------")
        print("End testing")
        print("")
        self.mdp.summary()

        
    def enterDefstates(self, ctx):
        states = [str(x) for x in ctx.ID()]
        self.mdp.states = states
        print("States defined by the user : %s" % str(states))

    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.mdp.actions = actions
        print("Actions defined by the user : %s" % str(actions))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))

        for i in range(len(weights)):
            if weights[i] < 0: weights[i] == 0
        
        if sum(weights) <= 0:
            print("WARNING: Weights for this action are equal to 0")
            print("This action will not be added")
            return


        if dep not in self.mdp.states:
            print(f"WARNING : The state {dep} was not defined, it's added automatically")
            self.mdp.states.append(dep)
        
        for id in ids:
            if str(id) not in self.mdp.states:
                print(f"WARNING : The state {str(id)} was not defined, it's added automatically")
                self.mdp.states.append(str(id))
            if str(id) not in self.mdp.accessible:
                self.mdp.accessible.append(str(id))

        if act not in self.mdp.actions:
            print(f"WARNING : The action {act} was not defined, it's added automatically.")
            self.mdp.actions.append(act)

        if act not in self.mdp.transition.keys():
            self.mdp.transition[act] = np.zeros((len(self.mdp.states), len(self.mdp.states)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[act][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight

        if dep not in self.mdp.possible_actions.keys():
            self.mdp.possible_actions[dep] = []

        self.mdp.possible_actions[dep].append(act)

        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

        for i in range(len(weights)):
            if weights[i] < 0: weights[i] == 0
        
        if sum(weights) <= 0:
            print("WARNING: Weights for this action are equal to 0")
            print("This action will not be added")
            return

        if dep not in self.mdp.states:
            print(f"WARNING : The state {dep} was not defined, it's added automatically")
            self.mdp.states.append(dep)

        for id in ids:
            if str(id) not in self.mdp.states:
                print(f"WARNING : The state {str(id)} was not defined, it's added automatically")
                self.mdp.states.append(str(id))
            if str(id) not in self.mdp.accessible:
                self.mdp.accessible.append(str(id))

        if None not in self.mdp.transition.keys():
            self.mdp.transition[None] = np.zeros((len(self.mdp.states), len(self.mdp.states)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[None][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight

        if dep not in self.mdp.possible_actions.keys():
            self.mdp.possible_actions[dep] = []

        self.mdp.possible_actions[dep].append(None)

    @property
    def getMDP(self):
        return self.mdp

            

def main():
    lexer = gramLexer(FileStream("ex2.mdp"))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    

    mdp = printer.getMDP
#    print(mdp.transition[None])
#    print(np.sum(mdp.transition[None], axis = 1))
#    print(mdp.possible_actions)
    mdp.print()
    mdp.simulate()
#    input("Press Enter to end program")


if __name__ == '__main__':
    main()
