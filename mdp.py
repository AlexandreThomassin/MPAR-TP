from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import numpy as np
import scipy as sp
import random

import pydot
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)

class MDP():
    def __init__(self):
        """
        Transition : Dictionnaire d'actions, chaque action a sa propre matrice de transition
        """
        # List of states
        self.states = []

        self.reward = []

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
        
        if sorted(self.accessible) == sorted(self.states):
            print("Tous les états sont accessibles")
        else:
            print(f"Les états suivants ne sont pas accessible : {list(set(self.states) - set(self.accessible))}")

        print("----------------------------------")

    def MDP_test(self):

        issue = False

        
        for state in self.states:
            index = self.states.index(state)

            # Test if a state has at least one transition, if not we add a loop on itself
            if state not in self.possible_actions.keys():
                    issue = True
                    print(f"WARNING : The state {state} has no transitions to other state.")
                    print("By default a transition on himself is added.\n")
                    if None not in self.transition.keys():
                        self.transition[None] = np.zeros((len(self.states), len(self.states)))

                    self.transition[None][index][index] = 1
                    self.possible_actions[state] = [None]

            # Test if state has transition with actions and without actions
            if None in self.possible_actions[state] and len(self.possible_actions[state]) != 1:
                        issue = True
                        print(f"WARNING : The state {state} has transition both with actions and without actions !")
                        print("When simulate it will choose the path without actions.\n")
                    

        # Resize transition matrix in case it is not of the right size
        for act in self.transition.keys():
            if len(self.transition[act]) != len(self.states):
                diff = abs(len(self.transition[act]) != len(self.states))
                self.transition[act] = np.pad(self.transition[act], ((0,diff),(0,diff)))

        # if len(self.transition[None]) != len(self.states):
        #     diff = abs(len(self.transition[None]) != len(self.states))
        #     self.transition[None] = np.pad(self.transition[None], ((0,diff),(0,diff)))
        
        # Test if there are non-accessible states
        if set(self.accessible) != set(self.states):
            issue = True
            print(f"WARNING : Les états suivants ne sont pas accessible : {list(set(self.states) - set(self.accessible))}\n")

        if not issue:
            print("Il n'y a pas de problèmes")

        # Get probability between 0 and 1
        for act in self.transition.keys():
            for i in range(len(self.states)):
                if np.sum(self.transition[act][i]) != 0:
                    self.transition[act][i] = self.transition[act][i] / np.sum(self.transition[act][i])

    def print(self):
        self.graph = pydot.Dot('Markov Chain Representation', graph_type='graph', bgcolor='white')
        try:
            states_graph = [pydot.Node(self.states[i], label = f"{self.states[i]} - [{self.reward[i]}]") for i in range(len(self.states))]
        except:
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
        #plt.pause(0.001)
        plt.ion()
        plt.show()
        return

    def update(self, c_state):
        plt.clf()

        if self.blue_state in self.states:
            self.graph.add_node(pydot.Node(self.blue_state, label = self.blue_state, style = ""))
        if c_state in self.states:
            self.graph.add_node(pydot.Node(c_state, label = c_state, style = "filled", fillcolor = 'cyan'))
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
        plt.pause(0.2)
        plt.ion()
        plt.show()


    def simulate(self, max_steps = 100):

        # If there are no states, end the simulation right away
        if len(self.states)==0:
            print("Il n'y pas d\'états, simulation impossible")
            return


        # Get the parameters of the simulation
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

            # Deterministic part
            if None in self.possible_actions[state]:

                # Get possible targets states and respectives weights
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]

                # Ending the simulation if we loop
                if len(targets) == 1 and targets[0] == state:
                    print("This state loop on himself, ending simulation.")
                    break
                
                # Choose the next state with the choices function
                choice = random.choices(targets, weights = weights)
                print(str(step) + ": Transition from " + state + " to " + choice[0])
                state = choice[0]
                

            # Non-deterministic part
            else:
                
                # If we loop on the state with every action, end the simulation
                loop = True
                for act in self.possible_actions[state]:
                    targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                    if not (len(targets) == 1 and targets[0] == state):
                         loop = False
                       
                if loop:
                    print("This state loop on himself, ending simulation.")
                    break
                
                # If automatic simulation is disabled, asked the user for the next action
                if auto == "N" :
                    act = input("Choisissez une action parmi les suivantes : " + str(self.possible_actions[state]) + "\n")
                    while act not in self.possible_actions[state]:
                        print("L'action ne fais pas parti des actions possible, veuillez ressayez")
                        act = input("Choisissez une action parmi les suivantes : " + str(self.possible_actions[state]) + "\n")
                
                # If automatic simulation is enabled, choose a random action
                else:
                    act = random.choice(self.possible_actions[state])

                # Get target states and respective weights
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
                weights = [self.transition[act][index][i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]

                choice = random.choices(targets, weights = weights)
                print(str(step) + ": Transition from " + state + " to " + choice[0] + " using action " + act)
                state = choice[0]
        if animate == 'Y':
            self.update(state)


    # Simulation function for Statistical model-checking
    def simu(self, goal, max_steps = 100):
        if len(self.states)==0:
            return

        state = self.states[0]

        for _ in range(max_steps):

            index = self.states.index(state)

            targets = []
            weights = []

            if None in self.possible_actions[state]:
                targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
                weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]

                if len(targets) == 1 and targets[0] == state:
                    break

                choice = random.choices(targets, weights = weights)
                state = choice[0]

                if state == goal:
                    return state
        
        return state
    
    # Monte-Carlo method for SMC
    def MonteCarlo(self, state, max_step, eps, sigma):
        N = int(np.ceil((np.log(2)-np.log(sigma))/((2*eps)**2)))
        proba = 0
        for _ in range(N):
            sim = self.simu(state, max_step)
            if sim == state:
                proba += 1

        return proba/N

    # SPRT method for SMC
    def SPRT(self, state, max_step, alpha, beta, theta, eps, N):
        gamma1 = theta - eps
        gamma0 = theta + eps
        dm = 0
        m = 0
        logA = np.log((1 - beta)/alpha)
        logB = np.log(beta/(1 - alpha))
        logRm = 0
        for _ in range(N):
            sim = self.simu(state, max_step)
            if sim == state:
                logRm += np.log(gamma1) - np.log(gamma0)
            else:
                logRm += np.log(1 - gamma1) - np.log(1 - gamma0)
            if logRm >= logA:
                return f"On valide l'hypothèse P(♦{state}) <= {theta} - {eps}"
            if logRm <= logB:
                return f"On valide l'hypothèse P(♦{state}) >= {theta} + {eps}"
        return False
    
    # Iter values algorithm to get opponent with best rewards
    def iter_values(self, gamma, epsilon, sens = "max"):

        # Stop immediatly if there are no rewards defined
        if not self.reward :
            print("This MDP has no reward")
            return (None,None)

        # Setup the itervalues algorithm to get max or min rewards
        if sens not in ["max", "min"]:
            raise ValueError("sens is either max or min")
        

        # Initialize variables
        V = np.zeros(len(self.states))
        new_V = np.zeros(len(self.states))
        flag = False
        #print(self.reward)


        while not flag:
            for i in range(len(new_V)):
                state = self.states[i]
                actions = self.possible_actions[state]
                if sens == "max":
                    maxi = max([sum([V[j]*self.transition[a][i][j] for j in range(len(self.states))]) for a in actions])
                elif sens == "min":
                    maxi = min([sum([V[j]*self.transition[a][i][j] for j in range(len(self.states))]) for a in actions])
                new_V[i] = self.reward[i] + gamma * maxi
            if np.linalg.norm(new_V - V) < epsilon:
                flag = True
            V = new_V.copy()
        sigma = [None]*len(self.states)

        # Get the opponent
        for s in self.states:
            i = self.states.index(s)
            actions = self.possible_actions[s]
            maxi = [sum([V[j]*self.transition[a][i][j] for j in range(len(self.states))]) for a in actions]
            if sens == "max":
                arg = np.argmax([self.reward[i] + gamma * maxi[a] for a in range(len(actions))])
            elif sens == "min":
                arg = np.argmin([self.reward[i] + gamma * maxi[a] for a in range(len(actions))])
            sigma[i] = actions[np.argmax([self.reward[i] + gamma * maxi[a] for a in range(len(actions))])]

        return V, sigma

    # Simulation function for Q-Learning algorithm, this function simulate only the next step
    def Q_simu(self, state, act):

        index = self.states.index(state)

        targets = []
        weights = []

        targets = [self.states[i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]
        weights = [self.transition[act][index][i] for i in range(len(self.states)) if self.transition[act][index][i] != 0]


        choice = random.choices(targets, weights = weights)
        state = choice[0]
        index = self.states.index(state)

        return state
    
    # Q-Learning algorithm
    def Q_Learning(self, T_tot, gamma):

        # Stop if this is a Markov Chain
        for state in self.states:
            if None in self.possible_actions[state]:
                print("Cannot apply Q-Learning on Markov Chain")
                return None

        # Initialize Variable
        Q = np.zeros((len(self.states), len(self.actions)))
        alpha = np.ones((len(self.states), len(self.actions)))
        #print(Q)
        state = self.states[0]
        index_state = self.states.index(state)


        for t in range(T_tot):
            act = random.choice(self.possible_actions[state])
            new_state = self.Q_simu(state, act)

            index_new_state = self.states.index(new_state)
            index_act = self.actions.index(act)

            alpha[index_state][index_act]+=1

            # Update de la fonction Q
            delta = self.reward[index_state] + gamma*max(Q[index_new_state]) - Q[index_state][index_act]
            Q[index_state][index_act] += (1/alpha[index_state][index_act])*delta

            state = new_state
            index_state = self.states.index(state)

        return Q

    # Probabilistic Model checking for Markov Chain
    def modelcheckMC(self, final_state, n = None):
        """
        Calcule P(♦s)
        """

        # Change function if the user tries to apply this function to MDP instead of Markov Chain
        for state in self.states:
            if not self.possible_actions[state] == [None]:
                print("WARNING: You are trying to use the modelcheckMC with a MDP, changing to modelcheckMDP")
                return self.modelcheckMDP(final_state, n)
            
        # Initialize S1 with the states given by the user
        if type(final_state) is list:
            S1 = final_state
        else:
            S1 = [final_state]

        # Initialize A and S0
        A = np.array(self.transition[None].copy())
        S0 = [self.states[s] for s in range(len(self.states)) if self.states[s] not in S1 and A[s][s] >= 1.]

        # Fill S1 and S0 with additional states if there are
        for state in self.states:
            index = self.states.index(state)
            targets = [self.states[i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]
            weights = [self.transition[None][index][i] for i in range(len(self.states)) if self.transition[None][index][i] != 0]

            if len(targets) == 1 and weights[0] == 1:
                if targets[0] in S1 and state not in S1:
                    S1.append(state)
                if targets[0] in S0 and state not in S0:
                    S0.append(state)

        
        print("S1 :" + str(S1))
        print("S0 :" + str(S0))

        # Get indexes of S1 and S0 states
        index_S1 = [self.states.index(s) for s in S1]
 
        index_S0 = [self.states.index(s) for s in S0]


        # Delete S0 and S1 lines from A and create b
        A = np.delete(A, index_S0 + index_S1, axis = 0)
        b = np.take(A, index_S1, axis = 1)
        b = np.sum(b, axis = 1)
        b = b.reshape((A.shape[0], 1))
        A = np.delete(A, index_S0 + index_S1, axis = 1)
        M = np.eye(A.shape[0]) - A

        # Compute gamma_n
        if n is None:
            # Solve Ax = b
            y = np.linalg.solve(M,b)
        else:
            y = np.zeros((A.shape[0],1))
            for _ in range(n):
                y = np.dot(A, y) + b
        return y

    # Probabilistic Model checking for Markovian decision process
    def modelcheckMDP(self, final_state):

        # Initialize S1 and S0
        S1 = [final_state]

        S0 = []

        # Add states that loop on themselves to S0
        for state in self.states:
            loop = True
            for act in self.transition.keys():
                index = self.states.index(state)
                if self.transition[act][index][index] != 1:
                    loop = False
                
            if loop == True and state not in S1:
                S0.append(state)

        # print("S1 :" + str(S1))
        # print("S0 :" + str(S0))

        # Get S0 and S1 states indexes
        index_S0 = [self.states.index(S) for S in S0]
        
        index_S1 = [self.states.index(S) for S in S1]


        len_S = len(self.states) - len(S0) - len(S1)
        S = np.delete(self.states,index_S0+index_S1)
        # print(len_S, len(S))

        total_lines = sum([len(self.possible_actions[s])+2 for s in S])

        # Define A and b
        A = np.zeros((total_lines, len_S))
        b = np.zeros((total_lines))

        # print(A.shape)
        # print(b.shape)
        

        # Fill and A and b 
        i = 0
        j = 0
        for state in S:
            index = self.states.index(state)
            for act in self.possible_actions[state]:
                A[i] = -np.delete(self.transition[act][index], index_S0+index_S1)
                A[i][j]+=1
                b[i] = np.sum(np.take(self.transition[act][index],index_S1))
                i+=1

            A[i][j] = 1
            
            i+=1
            A[i][j] = -1
            b[i] = -1
            i+=1
            j+=1
        
        
        c = np.ones(A.shape[1])

        # print(-c)

        # Min probability
        min_proba = sp.optimize.linprog(-c, A, b).x

        # Max probability
        max_proba = sp.optimize.linprog(c,-A,-b).x

        return (min_proba,max_proba)

    ### Reinforcement Learning

    # Main function
    def SMC4MDP(self, state, h, eps, N, L, p, eta, SPRT_max_step, SPRT_alpha, SPRT_beta, SPRT_theta, SPRT_eps, SPRT_N):
        # Compute T 
        T = np.log2(eta)/np.log2(1 - p)
        for _ in range(int(T)):
            sigma = np.zeros((len(self.states), len(self.actions)))

            for i in range(len(self.states)):
                for action in self.possible_actions[state]:
                    j = self.actions.index(action)
                    sigma[i][j] = 1/len(self.possible_actions[self.states[i]])
            sigma = self.optimise_sigma(state, sigma, h, eps, N, L)
            adversaire = self.determinise(sigma)

            if not self.hypothesisTesting(adversaire, state, SPRT_max_step, SPRT_alpha, SPRT_beta, SPRT_theta, SPRT_eps, SPRT_N):
                return False
            
        return True                    
    
    # Optimising scheduler
    def optimise_sigma(self,state, sigma, h, eps, N, L):
        for _ in (range(L)):
            Q = self.sigma_evaluate(state, sigma, N)
            # print(Q)
            sigma = self.sigma_improve(sigma, h, eps, Q)
            #print("Q = \n", Q, "\n sigma =", sigma, "\nScheduler = \n", self.determinise(sigma.copy()))
        return sigma
    
    # Get deterministic scheduler
    def determinise(self, sigma):
        for i in range(len(self.states)):
            line = np.zeros(len(self.actions))
            line[np.argmax([sigma[i][j] for j in range(len(self.actions))])] = 1.0
            sigma[i] = line.copy()
        return sigma

    # Make a simulation using a chosen sigma
    def simu_sigma(self, sigma, state, max_step = 200):
        res = []
        for _ in range(max_step):

            index_state = self.states.index(state)

            # Choix de l'actions selon sigma
            actions = self.actions
            weights = sigma[index_state]

            try:
                action = random.choices(actions, weights)[0]
            except:
                print(weights)        
                raise ValueError
            res.append((state,action))

            # Choix du prochain sommet
            targets = [self.states[i] for i in range(len(self.states)) if self.transition[action][index_state][i] != 0]
            weights = [self.transition[action][index_state][i] for i in range(len(self.states)) if self.transition[action][index_state][i] != 0]


            try:
                choice = random.choices(targets, weights = weights)[0]
            except:
                print(weights)
                raise ValueError

            # Check if the choice state loop on himself with every actions
            index_choice = self.states.index(choice)
            loop = True
            for act in self.possible_actions[choice]:
                if self.transition[act][index_choice][index_choice] == 1:
                    loop = loop and True
                else:
                    loop = False
            
            if loop:
                res.append((choice, self.possible_actions[choice][0]))
                return res
            state = choice
            
        return res

    # Evaluate sigma regarding the chosen property
    def sigma_evaluate(self, final_state, sigma, N = 1000):
        R_plus = np.zeros((len(self.states), len(self.actions)))
        R_moins = np.zeros((len(self.states), len(self.actions)))
        Q = np.zeros((len(self.states), len(self.actions)))

        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                Q[i,j] = sigma[i,j]

        for i in range(N):
            state = self.states[0]
            res_simu = self.simu_sigma(sigma, state)
            final_state_simu = res_simu[-1][0]
            # print(res_simu)
            # print('Final state : ' + str(final_state_simu))
            for (state, action) in res_simu:
                index_state = self.states.index(state)
                index_action = self.actions.index(action)

                if final_state_simu == final_state:
                    R_plus[index_state][index_action] += 1
                else:
                    R_moins[index_state][index_action] += 1

                Q[index_state][index_action] = R_plus[index_state][index_action] / (R_plus[index_state][index_action] + R_moins[index_state][index_action])
        

        return Q
    
    # Use SPRT algorithm to test if trained scheduler verify the property
    def hypothesisTesting(self, adversaire, state, max_step, alpha, beta, theta, eps, N):
        gamma1 = theta - eps
        gamma0 = theta + eps
        logA = np.log((1 - beta)/alpha)
        logB = np.log(beta/(1 - alpha))
        logRm = 0
        
        for _ in range(N):
            sim = self.simu_sigma(adversaire, self.states[0], max_step)[-1][0]
            if sim == state:
                logRm += np.log(gamma1) - np.log(gamma0)
            else:
                logRm += np.log(1 - gamma1) - np.log(1 - gamma0)

            if logRm >= logA:
                return True
            
            if logRm <= logB:
                return False
        return False

    # Improve scheduler based on evaluation
    def sigma_improve(self, sigma, h, eps, Q):
        res = sigma
        for i in range(len(self.states)):
            astar = np.argmax([Q[i][a] for a in range(len(self.actions))])
            somme = np.sum([Q[i][a] for a in range(len(self.actions))])

            p = np.zeros(len(self.actions))
            if somme > 0:
                for j in range(len(self.actions)):
                    p[j] = eps * Q[i][j] / somme
            p[astar] += 1 - eps
            for j in range(len(self.actions)):
                res[i][j] = h * sigma[i][j] + (1 - h) * p[j]
        return res


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

        
    # def enterDefstates(self, ctx):
    #     states = [str(x) for x in ctx.ID()]
    #     if "<missing ID>" in states:
    #         raise ValueError("Missing ID ")
    #     self.mdp.states = states
    #     print("States defined by the user : %s" % str(states))

    def enterStatenoreward(self, ctx):
        states = [str(x) for x in ctx.ID()]
        if "<missing ID>" in states:
            raise ValueError("Missing ID ")
        self.mdp.states = states
        print("States defined by the user : %s" % str(states))
    
    def enterStatereward(self, ctx):
        states = [str(x) for x in ctx.ID()]
        reward = [int(str(x)) for x in ctx.INT()]
        if "<missing ID>" in states:
            raise ValueError("Missing ID ")
        self.mdp.states = states
        self.mdp.reward = reward
        print("States defined by the user : %s, cost defined by the user %s" % (str(states), str(reward)))


    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.mdp.actions = actions
        for act in actions: 
            self.mdp.transition[act] = np.zeros((len(self.mdp.states),len(self.mdp.states)))
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

        if len(self.mdp.transition[act]) != len(self.mdp.states):
            diff = abs(len(self.mdp.transition[act]) != len(self.mdp.states))
            self.mdp.transition[act] = np.pad(self.mdp.transition[act], ((0,diff),(0,diff)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[act][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight

        if dep not in self.mdp.possible_actions.keys():
            self.mdp.possible_actions[dep] = []

        if act in self.mdp.possible_actions[dep]:
            print("WARNING: A transition from the same state and with the same action already exists. Only the last transaction defined will be considered.")
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

        if len(self.mdp.transition[None]) != len(self.mdp.states):
            diff = abs(len(self.mdp.transition[None]) != len(self.mdp.states))
            self.mdp.transition[None] = np.pad(self.mdp.transition[None], ((0,diff),(0,diff)))

        for (id, weight) in zip(ids, weights):
            self.mdp.transition[None][self.mdp.states.index(dep), self.mdp.states.index(id)] = weight

        if dep not in self.mdp.possible_actions.keys():
            self.mdp.possible_actions[dep] = []

        self.mdp.possible_actions[dep].append(None)

    @property
    def getMDP(self):
        return self.mdp

def open(mdp_file):
    lexer = gramLexer(FileStream(mdp_file))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    return printer.getMDP            
    
def main():
    mdp = open("simu-mdp.mdp")
    #print(mdp.transition)
    #mdp.print()

    print(mdp.possible_actions)
    print(mdp.actions)

    # SMC = mdp.modelcheckMC("F", n = 5)
    # print("Model Checking Statitique : \n" + str(SMC))

    # MonteCarlo = mdp.MonteCarlo("F", max_step=5, eps=0.01, sigma=0.05)
    # print("Résultat avec la méthode de Monte-Carlo : " + str(MonteCarlo))

    # SPRT = mdp.SPRT("F", max_step=5, alpha=0.01, beta=0.01, theta=0.67, eps=0.01, N=10000)
    # print(SPRT)


    # mdp.simulate(max_steps=50)
    #proba = [mdp.MonteCarlo(f"T{i}", 5, 0.01, 0.01) for i in range(1,7)]
    #print(proba)
    #Hyps = [mdp.SPRT(f"T{i}", 5, 0.01, 0.01, 0.1, 0.01, 30_000) for i in range(1,7)]
    #print(Hyps)
    # reward, opponent = mdp.iter_values(0.5, 1)
    # Q = mdp.Q_Learning(10000, 0.5)
    # print(Q)

    res = mdp.SMC4MDP('W', h=0.5, eps=0.01, N=2000, L=30, p=0.5, eta=0.1, SPRT_max_step=100, SPRT_alpha=0.01, SPRT_beta=0.01, SPRT_eps=0.1, SPRT_N=30_000, SPRT_theta=0.12)
    print(res)
    #print(mdp.modelcheck("F", 10))
    input("Press Enter to end program")


if __name__ == '__main__':
    main()
