import os

with open("datasets_to_use.txt","r") as f:
	lines = f.readlines()

import sys
sys.path.append('/Users/joelhogg/Documents/disertation stuff/bench_datasets/EXPERIMENTATION/METHODS/DAG_GNN/DAG_from_GNN')

for path in lines:
	print(path)
	os.system(f'python3.9 choose_dataset.py "{path}"')
	os.system(f'python3.9 -m "DAG_from_GNN"')
print("finished!")