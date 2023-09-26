#-=-=-=-=-= FOR BAYES NETWORKS -=-=-=-=-=-=-=-=-
import bnlearn as bn
import argparse
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("filename",type=str)
    parser.add_argument("-modelname",type=str,default="child.bif")
    args=parser.parse_args()
    return args
args=parse_args()
fn = args.filename
modelname = args.modelname

from pgmpy.readwrite import BIFReader
import networkx as nx

reader = BIFReader(modelname)
model = reader.get_model()
ground_truth_graph = nx.DiGraph(model.edges())
ground_truth_adj_matrix = nx.to_numpy_array(ground_truth_graph)

## draw adj mat as graph to check it works:
# pos = nx.spring_layout(ground_truth_graph,k=5)
# nx.draw(ground_truth_graph, pos, with_labels=True)
# # show the plot
# import matplotlib.pyplot as plt
# plt.show()


#IMPORT ADJ-MAT PRED AND RUN SHD
from cdt.metrics import SHD
from numpy import genfromtxt
import pandas as pd

input_data = pd.read_csv((f'{fn}'), index_col=0)
header_row = input_data.columns.values.tolist()
labels = dict(zip(range(0,len(header_row)), header_row))
G = nx.DiGraph(input_data)
pred_adj_mat = nx.to_numpy_array(G)

#RUN SHD

tar = ground_truth_graph
pred = pred_adj_mat

shd_result = SHD(tar,pred)
print(f"SHD: {shd_result}")


