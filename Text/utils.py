# from sentence_transformers import SentenceTransformer, util 
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from numba import njit,prange
from argparse import ArgumentParser
from transformers import TrainingArguments, Trainer
from collections import defaultdict
from transformers import DataCollatorWithPadding
# from datasets import load_metric
# import evaluate
import pandas as pd
import pickle
import random
from tqdm import tqdm
import heapq

def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data has been saved to {file_path}')

def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print(f'Data has been loaded from {file_path}')
    return loaded_data





def get_dataset(ratio=0.7):

    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    shuffled_dataset = imdb["train"].shuffle(seed=42)
    split_index = int(ratio * len(shuffled_dataset))
    train_dataset = shuffled_dataset.select(range(split_index))
    
    val_dataset = shuffled_dataset.select(range(split_index, len(shuffled_dataset)))
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = imdb["test"].shuffle(seed=42).map(preprocess_function, batched=True)


    return tokenized_train,tokenized_val,tokenized_test


def get_similarity(dataset):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(dataset['text'], convert_to_tensor=True)
    similarity = model.similarity(embeddings,embeddings)
    similarity =similarity.detach().cpu().numpy()

    return similarity

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}



def get_accuracy(tokenized_train,tokenized_val,tokenized_test,mask,num_label=2):

    
    accuracy = evaluate.load("accuracy")

    import numpy as np


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    
    model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )
    
    # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_label)

    training_args = TrainingArguments(
                output_dir="text_classification",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=2,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                # push_to_hub=True,
            )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train.select(mask),
        eval_dataset= tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train() 

    # training_args = TrainingArguments(
    #             output_dir="text_classification",
    #             learning_rate=2e-5,
    #             per_device_train_batch_size=16,
    #             per_device_eval_batch_size=16,
    #             num_train_epochs=2,
    #             weight_decay=0.01,
    #             save_strategy="epoch",
    #             trust_remote_code=True
    #             # push_to_hub=True,
    #             )
  
    # trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=tokenized_train.select(mask),
    #         eval_dataset=tokenized_val,
    #         tokenizer=tokenizer,
    #         data_collator=data_collator,
    #         compute_metrics=compute_metrics,
    #         )
    
    # trainer.eval_dataset = tokenized_test

    return trainer.evaluate(tokenized_test)['eval_accuracy']
    


# @njit(fastmath=True,parallel=True)
def SS(similarity,r,c):

    N = len(similarity)

    pruned_universe = set()

    universe_sparse = np.ones(N)
    universe=list(range(N))
    

    while len(universe)> r* np.log2(N):
        U=random.sample(universe,int(r*np.log2(N)))
        universe = set(universe)
        for element in tqdm(U):
            universe.remove(element)
            universe_sparse[element] = 0

        U = set(U)
        pruned_universe=pruned_universe.union(U)

        universe_gain,_=facility_location(similarity=similarity,costs=np.ones(N),budget=N,ground_set=universe_sparse) # f(V)

        universe_u_gain = {} # f(V U u)
        u_gain = {} # f(u)

        temp_groud_set = np.zeros(N)

        for u in tqdm(U):
            # universe.add(u)
            universe_sparse[u] = 1
            universe_u_gain[u],_ = facility_location(similarity=similarity,costs=np.ones(N),
                                                     budget=N,ground_set=universe_sparse) # f(V)
            universe_sparse[u] = 0
            # universe.remove(u)
            temp_groud_set[u] = 1
            u_gain[u],_ = facility_location(similarity=similarity,costs=np.ones(N),budget=N,ground_set=temp_groud_set) # f(V)

            temp_groud_set[u] = 0

        lst = []
        for v in tqdm(universe):

            w=float('inf')
            
            # for u in graph.neighbors(v):
                
            for u in U:
                # universe_copy=universe.copy()
                # universe_copy.append(u)
                temp_groud_set[u] = 1
                temp_groud_set[v] = 1 
                local_gain = facility_location(similarity=similarity,costs=np.ones(N),
                                               budget=N,ground_set=temp_groud_set)[0]-u_gain[u] # f(v U u) -f(u)
                # print(local_gain)
                temp_groud_set[u] = 0
                temp_groud_set[v] = 0 

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
            universe_sparse[node] = 0
            # universe.re
        universe = list(universe)


        

    pruned_universe=pruned_universe.union(set(universe))

    ground_set = np.zeros(N)
    for element in pruned_universe:
        ground_set[element] = 1

    return ground_set    








    





    pass



@njit(fastmath=True,parallel=True)
def QS(similarity, costs, delta, budget):
    """
    Maximizes the ground set based on similarity and cost constraints.

    Args:
        similarity (np.ndarray): NxN matrix representing pairwise similarity scores between text elements.
        costs (np.ndarray): 1D array representing the cost associated with each text element.
        delta (float): A constant to regulate the minimum gain to cost ratio.
        budget (float): The available budget for selecting elements.

    Returns:
        np.ndarray: A binary array indicating which elements are selected in the ground set (1 if selected, 0 if not).
    """
    
    # Number of elements in the set
    N = len(similarity)
    print('Size of unpruned ground set',N)
    
    # Current objective value
    curr_obj = 0
    
    # Maximum similarity values for each element
    max_similarity = np.zeros(N)
    
    # Ground set to keep track of selected elements (0: not selected, 1: selected)
    ground_set = np.zeros(N)
    
    # Loop through all elements to consider them for the ground set
    for element in range(N):
        obj_val = 0
        
        # Calculate the objective value by updating the maximum similarity
        for i in prange(N):
            obj_val += max(max_similarity[i], similarity[i, element])

        # Gain is the increase in the objective value
        gain = obj_val - curr_obj
        
        # Check if the gain-to-cost ratio meets the threshold based on delta and budget
        if gain / costs[element] >= delta / budget * curr_obj:
            # Update the current objective value with the gain
            curr_obj += gain
            
            # Mark the element as selected in the ground set
            ground_set[element] = 1
            for i in range(N):
                max_similarity[i] = max(max_similarity[i], similarity[i, element])

    print('Size of pruned ground set',ground_set.sum())
    return ground_set


@njit(fastmath=True,parallel=True)
def facility_location(similarity,costs,budget,ground_set):
    # N= 25000
    N = len(similarity)

    max_obj = 0
    total_cost = 0
    solution_sparse = np.zeros(N)

    max_similarity = np.zeros(N)

    while total_cost < budget:

        max_element = -1
        obj_val = np.zeros(N)

        for element in prange(N):
            if solution_sparse[element] == 0 and ground_set[element] ==1 and costs[element]+total_cost <=budget:


                for i in range(N):
                    obj_val[element] += max(max_similarity[i],similarity[i,element])
         

        max_element = np.argmax(obj_val)

        if obj_val[max_element] == max_obj:
            break

        else:
            solution_sparse[max_element] = 1
            total_cost += costs[max_element]
            for i in range(N):
                max_similarity[i] = max(max_similarity[i],similarity[i,max_element])
            
            max_obj = obj_val[max_element]

    # print(max_obj)
    # print(solution_sparse.sum())
    return max_obj,solution_sparse