

from utils import *





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    # parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--budget", type= int , default= 100, help="Budget" )
    # parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    args = parser.parse_args()
    delta = args.delta

    train_dataset,val_dataset,test_dataset = get_dataset()

    similarity = get_similarity(train_dataset)


    costs = np.ones(len(similarity))
    N= len(similarity)

    df = defaultdict(list)

    for ratio in [0.05,0.10,0.15,0.20,0.25,0.3]:

        df['ratios'].append(ratio)

        budget = np.sum(costs)*ratio

        print(budget)

        df['budgets'].append(ratio)

        pruned_ground_set=QS(similarity, costs, delta, budget)
        obj_val_pruned,solution_pruned = facility_location(costs=costs,budget=budget,similarity=similarity,ground_set=pruned_ground_set)
        mask_pruned = np.where(solution_pruned==1)[0]

        df['Accuracy (Pruned)'].append(get_accuracy(train_dataset,val_dataset,test_dataset,mask_pruned))

        obj_val_unpruned,solution_unpruned = facility_location(costs=costs,budget=budget,similarity=similarity,ground_set=np.ones(N))
        mask_unpruned= np.where(solution_unpruned ==1)[0]


        df['Accuracy (Unpruned)'].append(get_accuracy(train_dataset,val_dataset,test_dataset,mask_unpruned))

        # break

    df = pd.DataFrame(df)

    df.to_pickle('IMDB')
    print(df)

    print(get_accuracy(train_dataset,val_dataset,test_dataset,mask = np.ones(N)))

    