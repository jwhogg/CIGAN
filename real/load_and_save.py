from cdt.data import load_dataset
import pandas as pd
import networkx as nx
data, graph = load_dataset('sachs')
data.to_csv("sachs_data.csv")

graph_adj_matrix = nx.to_numpy_matrix(graph)
graph_df = pd.DataFrame(graph_adj_matrix, index=graph.nodes(), columns=graph.nodes())
graph_df.to_csv("sachs_target.csv")

print("finished!")
