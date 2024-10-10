import os
from collections  import defaultdict
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    # print(f'Data has been loaded from {file_path}')
    return loaded_data


# matric = 'TimeToPrune'
matric = 'QueriesToPrune'
for problem in [
                'MaxCover',
                # 'MaxCut',
                # 'IM'
              ]:
    root_folder=os.path.join(problem,'data')
    # datasets=['Facebook','DBLP','Skitter','YouTube']
    
    datasets=os.listdir(root_folder)
    print(datasets)
    df = defaultdict(list)
    for dataset in datasets:
        

        

        # print('*'*20)
        # print(dataset)
        dataset_path = os.path.join(root_folder,dataset)
        algorthims = os.listdir(dataset_path)

        # df ={'algorithm':[],'Size of Ground Set':[],'Ratio':[],'Queries':[]}
        
        # for algorthim in ['Quickfilter','SS','LeNSE','CombHelperTeacher','CombHelperStudent','GNNpruner']:
        for algorthim in ['SS','Quickfilter']:
          try:
            df_ = load_from_pickle(os.path.join(dataset_path,algorthim))
            df['algorithm'].append(algorthim)
            df['dataset'].append(dataset)
            df[f'{matric}'].append(df_[f'{matric}'].iloc[0])


          except:
            pass
    df = pd.DataFrame(df)
    # df['Queries'] = df['Queries'].apply(lambda x: f"{x:.4f}")
    # df['Size of Ground Set']=df['Size of Ground Set'].round(4) 
    # df['Queries'] = df['Queries'].round(4)

    dataset_order = ['Facebook', 'Wiki','Deezer','Slashdot','Twitter','DBLP','YouTube',
                     'Skitter'
                     ]
    # Convert the dataset column to a categorical type with the specified order
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    print(df)
    # print('-'*20)

    # Pivot the dataframe to have 'SS' and 'Quickfilter' as columns
    df_pivot = df.pivot(index='dataset', columns='algorithm', values=f'{matric}')

    # Calculate the ratio of Quickfilter/SS
    df_pivot['Ratio'] =  df_pivot['SS']/df_pivot['Quickfilter'] 

    # Reset the index for plotting
    df_pivot.reset_index(inplace=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Assuming df_pivot is already defined
    plt.figure(dpi=300, figsize=(8, 4))
    ax = sns.barplot(x='dataset', y='Ratio', data=df_pivot, 
                     palette=sns.color_palette("Spectral"),
                     edgecolor='black')

    # # Set a single hatch pattern (e.g., '/' or 'x') for all bars
    # hatch = '/'  # Change this to any desired hatch pattern
    # edge_color = 'black'  # Color for the edges

    # # Apply the same hatch and edge color to each bar
    # for bar in ax.patches:
    #     bar.set_hatch(hatch)
    #     bar.set_edgecolor(edge_color)  # Set the edge color
    #     bar.set_linewidth(1.5)  # Optional: Set the linewidth of the edges

    # Adding the value labels on top of the bars
    for bar in ax.patches:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval,  # Position the text at the height of the bar
            f'{yval:.0f}',  # Format the value to 2 decimal places
            ha='center',  # Center the text horizontally
            va='bottom',  # Position the text just above the bar
            fontsize=16,  # Optional: Font size for the labels
            color='black'  # Optional: Color for the labels
        )

    # Set the labels and title
    plt.xlabel('')
    plt.ylabel('Queries (SS/QP)', fontsize=20)
    plt.xticks(rotation=65, fontsize=20)
    plt.yticks(fontsize=20)
    # Turn off y-axis ticks and labels
    plt.yticks([])

    # Adjust the layout and remove the top and right spines
    plt.tight_layout()
    sns.despine()

    plt.savefig(f'{problem}_{matric}.pdf', bbox_inches='tight', dpi=300)
    # Show the plot
    # plt.show()
    break
