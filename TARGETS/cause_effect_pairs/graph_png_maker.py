import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

with open("datasets_to_use.txt","r") as f:
	lines = f.readlines()

for file in lines:
	result = pd.read_csv(file[0:-1], index_col=0)
	# print(result)
	# G = nx.DiGraph(result.to_numpy())
	G = nx.from_pandas_adjacency(result, create_using=nx.DiGraph)

	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True)

	# Save the graph as a PNG image file
	short = file.split("/")[-1]
	plt.savefig(f"plots/{short}.png")
	plt.clf()
	result, G, pos, short = None, None, None, None

print("finished!")