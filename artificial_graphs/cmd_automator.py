import os

samples = [5000]
nodes = [10,20,50,100]
# graph_types = ['linear','polynomial','sigmoid_add','gp_add','nn','linear_ER']
graph_types = ['gp_add','nn','linear_ER']


for graph_type in graph_types:
	for node in nodes:
		for sample_size in samples:
			if ((node==10) and (graph_type=='gp_add')): #remove this
				continue #								remove this also
			os.system(f'python3.9 graph_generator.py {graph_type} -nodes={node} -sample_size={sample_size}')
			print(f"done: {graph_type}, {node} nodes, {sample_size} samples")

print("finished!")


