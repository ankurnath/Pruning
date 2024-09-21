from transformers import AutoFeatureExtractor, AutoModel
from collections import defaultdict
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from numba import njit,prange
import random
from tqdm import tqdm
import heapq

def calculate_obj(candidate_similarity,query_similarity,solution):

    obj_val = 0
    for candidate in solution:
        obj_val += 10 * np.sum(query_similarity[:,candidate])


    solution = list(solution)

    for  i in range(len(solution)):
        for j in range(i+1,len(solution)):
            obj_val -= candidate_similarity[i,j]

    return obj_val

def SS(candidate_similarity,query_similarity,r=8,c=8):

    n = len(candidate_similarity)
    pruned_universe=set()

    universe = list(range(n))

    while len(universe)> r*np.log2(n):
        U=random.sample(universe,int(r*np.log2(n)))
        
        universe = set(universe)

        for element in tqdm(U):
            universe.remove(element)

        U = set(U)
        pruned_universe=pruned_universe.union(U)
        universe_gain=calculate_obj(candidate_similarity,query_similarity,universe) # f(V)

        universe_u_gain = {} # f(V U u)
        u_gain = {} # f(u)
        # get all neighbors 
        
        
        for u in tqdm(U):
            universe.add(u)
            universe_u_gain[u] = calculate_obj (candidate_similarity,query_similarity ,universe)
            universe.remove(u)
            u_gain[u] = calculate_obj (candidate_similarity,query_similarity, [u])


        lst = []

        for v in tqdm(universe):

            w=float('inf')
            
            # for u in graph.neighbors(v):
                
            for u in U:
                # universe_copy=universe.copy()
                # universe_copy.append(u)
                
                local_gain = calculate_obj(candidate_similarity,query_similarity,[u,v])-u_gain[u] # f(v U u) -f(u)
                # print(local_gain)

                global_gain = universe_u_gain[u]-universe_gain
                w=min(w,local_gain-global_gain)

            lst.append((w,v))

        remove_nodes=heapq.nsmallest(int((1-1/np.sqrt(c))*len(universe)), lst)
        # print(remove_nodes)
        universe = set(universe)
        for w,node in tqdm(remove_nodes):
            # if w>0:
            #     print(w)
            universe.remove(node)
            # universe.re
        universe = list(universe)
        

        

    pruned_universe=pruned_universe.union(set(universe))

    ground_set = np.zeros(n)
    for element in pruned_universe:
        ground_set[element] = 1

    print('Size of pruned ground set(SS)', ground_set.sum())

    return ground_set







@njit(fastmath=True, parallel=True)
def QS(candidate_similarity, query_similarity, delta=0.05, budget=5):
    N = len(candidate_similarity)
    print('Size of unpruned ground set', N)

    # Current objective value
    curr_obj = 0

    # Ground set, dense and sparse solutions
    ground_set = np.zeros(N)
    solution_dense = np.zeros(N, dtype=np.int32)
    solution_sparse = np.zeros(N)
    size_solution = 0

    for element in range(N):
        gains = calculate_gains(candidate_similarity, query_similarity, 
                                solution_dense, size_solution, solution_sparse)

        gain = gains[element]

        if gain >= delta / budget * curr_obj:
            curr_obj += gain
            # Mark the element as selected in the ground set
            ground_set[element] = 1
            solution_sparse[element] = 1
            solution_dense[size_solution] = element
            size_solution += 1

    print('Size of pruned ground set(QS)', ground_set.sum())
    return ground_set




@njit(fastmath=True, parallel=True)
def calculate_gains(candidate_similarity, query_similarity, solution_dense, size_solution, c=2):
    """
    Calculate the gains for each candidate based on the current solution.

    Args:
    candidate_similarity (np.ndarray): A 2D array representing candidate-candidate similarity matrix.
    query_similarity (np.ndarray): A 2D array representing candidate-query similarity matrix.
    solution_dense (np.ndarray): Indices of selected candidates in the solution.
    size_solution (int): Number of selected candidates in the solution.
    c (float): Constant multiplier for query similarity.

    Returns:
    np.ndarray: An array of gains for each candidate.
    """
    num_candidates = len(candidate_similarity)
    gains = np.zeros(num_candidates)

    for candidate in prange(num_candidates):
        # print('candidate',candidate)
        # print(np.sum(query_similarity[:,candidate]) *2)
        # print(candidate_similarity[candidate, candidate])
        gains[candidate] = 10 * np.sum(query_similarity[:,candidate]) - candidate_similarity[candidate, candidate]

        # print(gain)

        # Adjust gain based on current solution
        for selected_idx in range(size_solution):
            selected_candidate = solution_dense[selected_idx]
            gains[candidate] -= 2 * candidate_similarity[candidate, selected_candidate]
        # gains[candidate] = gain

    return gains

# @njit(fastmath=True, parallel=True)
def graph_cut(candidate_similarity, query_similarity, budget, ground_set):
    """
    Perform the graph cut optimization to maximize similarity between candidates and the query.

    Args:
    candidate_similarity (np.ndarray): A 2D array representing candidate-candidate similarity matrix.
    query_similarity (np.ndarray): A 2D array representing candidate-query similarity matrix.
    budget (int): The number of candidates to select.
    ground_set (list or np.ndarray): Set of candidate elements.

    Returns:
    obj_val (float): The final objective value achieved.
    solution_dense (np.ndarray): Indices of selected candidates.
    solution_sparse (np.ndarray): Sparse solution with binary representation (selected candidates are marked 1).
    """

    num_candidates = len(candidate_similarity)

    # Initialize variables
    solution_sparse = np.zeros(num_candidates)
    solution_dense = np.zeros(num_candidates, dtype=np.int32)
    size_solution = 0
    obj_val = 0.0
    

    # Loop over the budget to select candidates
    for _ in range(budget):
        # Calculate gains for all candidates
        gains = calculate_gains(candidate_similarity, query_similarity, solution_dense, size_solution)

        best_candidate = -1
        max_gain = -float('inf')  # Initialize max_gain to negative infinity to ensure the first gain is selected
        
        for i in range(num_candidates):
            if ground_set[i] == 1:  # Check if candidate is eligible (not already selected)
                if gains[i] > max_gain and solution_sparse[i] == 0:  # Find the candidate with the maximum gain
                    max_gain = gains[i]
                    best_candidate = i

        if best_candidate == -1:
            # No valid candidate was found, stop the loop
            print("No valid candidate found, breaking the loop.")
            break

        # Select the candidate with the highest gain
        solution_dense[size_solution] = best_candidate
        solution_sparse[best_candidate] = 1
        size_solution += 1
        obj_val += gains[best_candidate]


    return obj_val, solution_dense[:size_solution], solution_sparse


def show_images(images, cols=5, titles=None):
    """
    Display a list of PIL images in a grid.
    
    Args:
        images (list): A list of PIL Image objects to display.
        cols (int): Number of columns in the grid (default is 3).
        titles (list): Optional list of titles for each image (must match the number of images).
    """
    num_images = len(images)
    rows = ceil(num_images / cols)
    
    # Create subplots with dynamic number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=(50, 10 * rows))
    if titles:
        plt.suptitle(titles,fontsize = 40)
    # Flatten the axes array for easy indexing, if needed
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')  # Remove axis
            # if titles and i < len(titles):
            #     ax.set_title(titles[i], fontsize=12)
        else:
            # Hide any unused subplot axes
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""


    emb_one_normalized = F.normalize(emb_one, p=2, dim=1).double()
    emb_two_normalized = F.normalize(emb_two, p=2, dim=1).double()
    scores= torch.mm(emb_one_normalized, emb_two_normalized.t())
    scores += 1
    scores /=2
    return scores.numpy()

def extract_embeddings(model: torch.nn.Module,transformation_chain):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp