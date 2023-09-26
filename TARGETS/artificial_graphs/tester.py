#FOR ARTIFICIAL GRAPHS

import argparse
import pandas as pd
import networkx as nx
from cdt.metrics import SHD
import numpy as np
import networkx as nx


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("-target_path",type=str)
    parser.add_argument("-result_path",type=str)
    args=parser.parse_args()
    return args

args=parse_args()
result_path = args.result_path
target_path = args.target_path

result = pd.read_csv(result_path, index_col=0)
# print(result.to_numpy())
G_result = nx.DiGraph(result.to_numpy())
result_mat = nx.to_numpy_array(G_result)

# target = pd.read_csv(target_path, index_col=0)
# G_target = nx.DiGraph((target.to_numpy()))
# target_mat = nx.to_numpy_array(G_target)

target = pd.read_csv(target_path, index_col=0)
target_with_index = np.c_[target.index.values, target.values]
G_target = nx.DiGraph(target_with_index)
target_mat = nx.to_numpy_array(G_target)

tar = target_mat
pred = result_mat

shd_result = SHD(tar,pred)
print(f"SHD: {shd_result}")

import csv
with open("SHD_results.txt","a") as f:
	writer = csv.writer(f)
	writer.writerow([result_path, shd_result])