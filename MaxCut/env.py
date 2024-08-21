
from utils import *
import random
import torch
from torch_geometric.utils.convert import  from_networkx
from env_utils import *
import numpy as np

from collections import defaultdict


class MaxCoverSystem(object):


    def __init__(self,
                 file_path =None,
                 observables = None,
                 seed = None,
                 budget = None,
                 ) -> None:
        

        if seed != None:
            np.random.seed (seed)


        
        assert observables[0] == Observable.STATE, "First observable must be Observation.STATE."
        self.observables = list(enumerate(observables))

        self.graph = load_from_pickle(file_path=file_path)
        self.n = self.graph.number_of_nodes()
        self.mapping = dict(zip(self.graph.nodes(), range(self.n)))
        self.step = 0
        self.sequence = list(range( self.n))
        self.budget = budget



    
    def get_gains(self,graph):
        return {node:graph.degree(node)+1 for node in graph.nodes()}

      
    def gain_adjustment(self,graph,gains,selected_element,uncovered):


        if uncovered[selected_element]:
            gains[selected_element]-=1
            uncovered[selected_element]=False
            for neighbor in graph.neighbors(selected_element):
                if neighbor in gains and gains[neighbor]>0:
                    gains[neighbor]-=1

        for neighbor in graph.neighbors(selected_element):
            if uncovered[neighbor]:
                uncovered[neighbor]=False
                
                if neighbor in gains:
                    gains[neighbor]-=1
                for neighbor_of_neighbor in graph.neighbors(neighbor):
                    if neighbor_of_neighbor  in gains:
                        gains[neighbor_of_neighbor ]-=1


    def reset(self):

        self.step = 1

        self.gain = self.get_gains(self.graph)

        self.uncovered = defaultdict(lambda: True)

        random.shuffle(self.sequence)
        self.state= torch.zeros((self.n,len(self.observables)))

        self.data = from_networkx(self.graph)

        for idx,obs in enumerate(self.observables):

            if obs == Observable.STATE:
                self.state[self.sequence[0],idx] = 1


    def step(self,action):

        done = False

        rew = 0
        if self.step > self.n:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError
    

        if action:
            # if action is true then keep the vertex or else remove it
            rew += self.gains[self.sequence[self.step]]

        else:
            self.state[self.sequence[self.step]] = 0


        self.step += 1

        if self.step == self.n:
            done = True

        else:
            self.state[self.sequence[self.step]] = 1
        return (self.get_observation(), rew, done, None)
    

    def train(self):
        self.testing=False
        self.training=True

    def test(self):
        self.training=False
        self.testing=True

    def get_observation(self):
        if self.training:
            return self.data.clone()
        else:
            return self.data


from argparse import ArgumentParser

        
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default='Facebook',
        help="Name of the dataset to be used (default: 'Facebook')"
    )

    args = parser.parse_args()

    file_path=f'../../data/train/{args.dataset}'

    env=MaxCoverSystem(file_path=file_path,
                   observables= [Observable.STATE],
                   budget=100)
    
    env.reset()




    


            



        



        

        