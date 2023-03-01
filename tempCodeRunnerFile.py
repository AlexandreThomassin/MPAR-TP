        for act in self.actions:
            if len(self.transition[act]) != len(self.states):
                diff = abs(len(self.transition[act]) != len(self.states))
                self.transition[act] = np.pad(self.transition[act], ((0,diff),(0,diff)))