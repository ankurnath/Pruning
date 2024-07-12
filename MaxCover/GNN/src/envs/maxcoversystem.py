
from utils import *
import random
import torch
from torch_geometric.utils.convert import  from_networkx



class MaxCoverSystem(object):


    def __init__(self,
                 file_path,
                 observables,
                 ) -> None:
        
        self.observables=list(enumerate(observables))

        self.graph= load_from_pickle(file_path=file_path)
        self.n = self.graph.number_of_nodes()
        self.mapping= dict(zip(self.graph.nodes(), range(self.n)))
        self.step=0
        self.sequence=list(range(0, self.n))

    def reset(self):

        self.step=0

        random.shuffle(self.sequence)

        self.state= torch.zeros((self.n,len(self.features)))

        self.data = from_networkx(self.graph)
        # self.data.x=torch.from_numpy(state)


        for idx,obs in enumerate(self.observables):
            pass

    def step(self,action):

        done = False

        rew =0

        self.step +=1

        if self.step > self.n:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError
        
        if action:
            
            pass



        else:
            pass


        return self.state,rew,done,None
    

    def train(self):
        self.testing=False
        self.training=True

    def test(self):
        self.training=False
        self.testing=True



        

    


            



        



        

        