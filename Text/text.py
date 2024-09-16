

from utils import *





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.1, help="Delta")
    parser.add_argument( "--r", type=float, default=8, help="r" )
    parser.add_argument( "--c", type=float, default=8, help="c" )
    # parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--budget", type= int , default= 100, help="Budget" )
    # parser.add_argument("--cost_model",type= str, default= 'degree', help = 'model of node weights')
    args = parser.parse_args()
    delta = args.delta
    r = args.r 
    c = args.c

    train_dataset,val_dataset,test_dataset = get_dataset()

    similarity = get_similarity(train_dataset)


    costs = np.ones(len(similarity))
    N= len(similarity)

    df = defaultdict(list)


    # pruned_ground_set_SS=SS(similarity,r,c)
   

    for ratio in [0.01,0.03,0.05,0.07,0.10]:

        

        budget = np.sum(costs)*ratio

        print(budget)
        

        print('Size of unpruned ground set',N)

        # obj_val_pruned_SS,solution_pruned_SS = facility_location(costs=costs,budget=budget,
        #                                                    similarity=similarity,
        #                                                    ground_set=pruned_ground_set_SS)
        
        # mask_pruned_SS = np.where(solution_pruned_SS==1)[0]
        # accuracy_pruned_SS = get_accuracy(train_dataset,val_dataset,test_dataset,mask_pruned_SS)
        # print('accuracy_pruned:',accuracy_pruned_SS)
        
        pruned_ground_set=QS(similarity, costs, delta, budget)
        print('Size of pruned ground set',pruned_ground_set.sum())
        obj_val_FS_QS,solution_pruned = facility_location(costs=costs,budget=budget,
                                                           similarity=similarity,
                                                           ground_set=pruned_ground_set)
        mask_pruned = np.where(solution_pruned==1)[0]
        print('Size of pruned training dataset(FS+QS)',len(mask_pruned))
        # df['Size of ']
        accuracy_FS_QS = get_accuracy(train_dataset,val_dataset,test_dataset,mask_pruned)
        print('accuracy(FS+QS):',accuracy_FS_QS)

        # random_mask= np.random.choice(np.arange(N), size=len(mask_pruned), replace=False)
        random_mask= np.random.choice(np.arange(N), size=224, replace=False)
        random_ground_set = np.zeros(N)
        random_ground_set[random_mask] = 1 
        obj_val_FS_RANDOM,solution_FS_RANDOM = facility_location(costs=costs,budget=budget,
                                                           similarity=similarity,
                                                           ground_set=random_ground_set)
        
        mask_FS_RANDOM = np.where(solution_FS_RANDOM==1)[0]
        
        accuracy_FS_RANDOM = get_accuracy(train_dataset,val_dataset,test_dataset,mask_FS_RANDOM)
        # accuracy_FS_RANDOM = get_accuracy(train_dataset,val_dataset,test_dataset,mask_pruned)
        print('accuracy(FS+Random):',accuracy_FS_RANDOM)
        break

        

    #     obj_val_unpruned,solution_unpruned = facility_location(costs=costs,budget=budget,
    #                                                            similarity=similarity,ground_set=np.ones(N))
    #     mask_unpruned= np.where(solution_unpruned ==1)[0]
        
        
    #     accuracy_unpruned = get_accuracy(train_dataset,val_dataset,test_dataset,mask_pruned)
    #     print('accuracy_unpruned:',accuracy_unpruned)

    #     df['Ratio'].append(ratio)
    #     df['Budget'].append(budget)
    #     df['Training Set Size'].append(N)
    #     df['Pruned Ground Set Size (FS from QS)'].append(pruned_ground_set.sum())
    #     df['Accuracy (FS + QS)'].append(accuracy_FS_QS)
    #     # df['Accuracy (FS + SS)'].append(accuracy_pruned_SS)
    #     df['Accuracy (FS + Random)'].append(accuracy_FS_RANDOM)
    #     df['Pruned Training Set Size (FS)'].append(len(mask_unpruned))
    #     df['Accuracy (FS)'].append(accuracy_unpruned)

    #     # break

    # df = pd.DataFrame(df)

    
    # print(df)

    # df['Accuracy'] = [get_accuracy(train_dataset,val_dataset,test_dataset,mask = np.arange(N))]*len(df)
    # df.to_pickle('IMDB')

    # print(df)

    