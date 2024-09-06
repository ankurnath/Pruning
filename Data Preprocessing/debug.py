from utils import *
import random

for dataset in ['Facebook',
                'Wiki',
                'Deezer',
                'Slashdot',
                'Twitter',
                'DBLP',
                'YouTube',
                'Skitter'
                ]:
    
    graph = load_graph(f'../../data/snap_dataset/{dataset}.txt')

    random.seed(0)

    node_weights = {node:random.random() for node in graph.nodes()}

    save_to_pickle(node_weights,f'../../data/snap_dataset/{dataset}_random')


# # # Open the file in read mode
# # with open(f'../../data/snap_dataset/Twitter.txt', 'r') as file:
# #     # Read and print the first 10 lines
# #     for i in range(10):
# #         line = file.readline()
# #         if line == '':
# #             break  # Stop if there are less than 10 lines in the file
# #         print(line.strip())


# import csv
# import pandas as pd
# # Define the input and output file paths
# input_file = f'../../data/snap_dataset/Deezer.csv'

# output_file = f'../../data/snap_dataset/Deezer.txt'
# df = pd.read_csv(input_file)

# with open(output_file, 'a') as f:
#     f.write(df.to_string(header=False, index=False))

# # print(df.head(10))


# # # Open the CSV file and read its contents
# # with open(input_file, 'r') as csv_file:
# #     csv_reader = csv.reader(csv_file)

# #     # Open the output file in write mode
# #     with open(output_file, 'w') as txt_file:
# #         # Iterate over each row in the CSV and write to the text file
# #         for row in csv_reader:
# #             # Join the row into a string separated by commas and write to the text file
# #             txt_file.write(', '.join(row) + '\n')

# # print(f"CSV data has been saved to {output_file}.")