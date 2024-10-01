# python knapsack_quickfilter.py --dataset Facebook
# python knapsack_quickfilter.py --dataset Wiki
# python knapsack_quickfilter.py --dataset Deezer
# python knapsack_quickfilter.py --dataset Slashdot

# python size_quickfilter.py --num_rr 1000000  --dataset Facebook  --budget 100 --delta 0.1 
# python size_quickfilter.py --num_rr 1000000  --dataset DBLP  --budget 100 --delta 0.1 
# python size_quickfilter.py --num_rr 1000000   --dataset Slashdot  --budget 100 --delta 0.1
# python size_quickfilter.py --num_rr 1000000  --dataset Twitter  --budget 100 --delta 0.1
# python size_quickfilter.py --num_rr 1000000  --dataset Deezer  --budget 100 --delta 0.1
# python size_quickfilter.py --num_rr 1000000   --dataset Wiki  --budget 100 --delta 0.1
# python size_quickfilter.py --num_rr 1000000   --dataset YouTube  --budget 100 --delta 0.1
# python size_quickfilter.py --num_rr 100000  --dataset Skitter  --budget 100 --delta 0.1 


# python GNN_pruning.py   --dataset Facebook  --budget 100  
# python GNN_pruning.py   --dataset DBLP  --budget 100  
# python GNN_pruning.py    --dataset Slashdot  --budget 100 
# python GNN_pruning.py   --dataset Twitter  --budget 100 
# python GNN_pruning.py   --dataset Deezer  --budget 100 
# python GNN_pruning.py    --dataset Wiki  --budget 100 
# python GNN_pruning.py    --dataset YouTube  --budget 100


# python Bilmes.py  --num_rr 1000000  --dataset Facebook  --budget 100  
# python Bilmes.py  --num_rr 1000000  --dataset DBLP  --budget 100  
# python Bilmes.py  --num_rr 1000000   --dataset Slashdot  --budget 100 
# python Bilmes.py  --num_rr 1000000  --dataset Twitter  --budget 100 
# python Bilmes.py  --num_rr 1000000  --dataset Deezer  --budget 100 
# python Bilmes.py  --num_rr 1000000   --dataset Wiki  --budget 100 
# python Bilmes.py  --num_rr 1000000   --dataset YouTube  --budget 100
# python Bilmes.py  --num_rr 1000000   --dataset Skitter  --budget 100 
python knapsack_gnn_pruning.py --dataset Facebook  --cost_model aistats
python knapsack_gnn_pruning.py --dataset Wiki  --cost_model aistats  
python knapsack_gnn_pruning.py --dataset Deezer  --cost_model aistats 
python knapsack_gnn_pruning.py --dataset Slashdot  --cost_model aistats 
python knapsack_gnn_pruning.py --dataset Twitter  --cost_model aistats  
python knapsack_gnn_pruning.py --dataset DBLP  --cost_model aistats  
python knapsack_gnn_pruning.py --dataset YouTube  --cost_model aistats  
python knapsack_gnn_pruning.py --dataset Skitter  --cost_model aistats


# python knapsack_multibudget.py --dataset Facebook 

# python knapsack_multibudget.py --dataset Wiki

# python knapsack_multibudget.py --dataset Deezer
# # python top_k_knapsack.py --dataset Deezer --budget 100
# python knapsack_multibudget.py --dataset Slashdot
# # python top_k_knapsack.py --dataset Slashdot --budget 100
# python knapsack_multibudget.py --dataset Twitter
# # python top_k_knapsack.py --dataset Twitter --budget 100
# python knapsack_multibudget.py --dataset DBLP
# # python top_k_knapsack.py --dataset DBLP --budget 100
# python knapsack_multibudget.py --dataset YouTube
# python top_k_knapsack.py --dataset YouTube --budget 100




# python size_quickfilter.py --dataset Facebook --budget 100 --delta 0.1
# python size_quickfilter.py --dataset Wiki --budget 100 --delta 0.1
# python size_quickfilter.py --dataset Deezer --budget 100 --delta 0.1
# python size_quickfilter.py --dataset Slashdot --budget 100 --delta 0.1
# python size_quickfilter.py --dataset Twitter --budget 100 --delta 0.1
# python size_quickfilter.py --dataset DBLP --budget 100 --delta 0.1
# python size_quickfilter.py --dataset YouTube --budget 100 --delta 0.1
# python size_quickfilter.py --dataset Skitter --budget 100 --delta 0.1