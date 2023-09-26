import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import pandas as pd
import networkx as nx
import datetime
from cdt.causality.graph import PC
from cdt.causality.graph import GES
from cdt.metrics import SHD
import networkx as nx
import re

import CIGAN7

batch_size = 50
latent_dim = 100
hidden_dim = 200
lr=0.0002 #0.0002
epochs = 200
sample_interval = 400

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        data_np = np.loadtxt(csv_path, dtype=np.float32, delimiter=',', skiprows=1)
        self.col_names = next(csv.reader(open(csv_path), delimiter=','))
        self.data = torch.from_numpy(data_np)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    def get_col_names(self):
        return self.col_names
    
# path = 'datasets/sachs_data_noindex.csv'

# model = CIGAN7.CIGAN(dataset, batch_size, latent_dim, hidden_dim, lr, epochs, sample_interval)
# model.set_debug(True)

# model.train()

# sample = model.get_synthetic_sample()
# print(sample)


### saves model as: model_name_[latent dim]_[hidden dim]_[lr]_[epochs].pt

class Tester:
    def __init__(self):
        self.results = {}
        self.test_results = {}
        self.ground_truths = {}
        self.batch_size=50
        self.model = None

    def set_test_range(self,latent_dim,hidden_dim,lr,epochs): #batch_size = (5,50), lr= (0.0002,0.5)
        # self.range_batch_size = np.linspace(batch_size[0],batch_size[1],num=25)
        self.range_latent_dim = np.linspace(latent_dim[0],latent_dim[1],num=5)
        self.range_hidden_dim = np.linspace(hidden_dim[0],hidden_dim[1],num=5)
        self.range_lr = np.linspace(lr[0],lr[1],num=5)
        # self.range_epochs = np.linspace(epochs[0],epochs[1],num=25)

    def set_dataset(self,path):
        self.dataset = CSVDataset(path)
    
    def run(self):
        for latent_dim in self.range_latent_dim:
            latent_dim=int(latent_dim)
            for hidden_dim in self.range_hidden_dim:
                hidden_dim=int(hidden_dim)
                for lr in self.range_lr:
                    # for epochs in self.range_epochs:
                    # epochs=int(epochs)
                    epochs=200
                    model = CIGAN7.CIGAN(self.dataset, self.batch_size, latent_dim, hidden_dim, lr, epochs, sample_interval)
                    model.train()
                    sample = model.get_synthetic_sample()
                    self.results[(latent_dim,hidden_dim,lr,epochs)] = sample
                    self.ground_truths[(latent_dim,hidden_dim,lr,epochs)] = model.get_adj_A()
                    torch.save(model.get_generator().state_dict(), (f'models/model__generator_{latent_dim}_{hidden_dim}_{lr}_{epochs}.pt'))
                    torch.save(model.get_discriminator().state_dict(), (f'models/model__discriminator_{latent_dim}_{hidden_dim}_{lr}_{epochs}.pt'))
                    print(f'done with latent_dim: {latent_dim} hidden_dim: {hidden_dim} lr: {lr} epochs: {epochs}')
    
    def test(self):
        for key in self.results:
            print(key) #key
            print(self.results[key]) #value
            # dag_gnn_result = self.test_dag_gnn(self.results[key],ground_truth)
            # self.test_results[key] = dag_gnn_result
            ground_truth = self.ground_truths[key]
            pc_result = self.test_pc(self.results[key],ground_truth)
            print(f'pc: {pc_result}')
            self.test_results[key] = pc_result

            ges_result = self.test_ges(self.results[key],ground_truth)
            self.test_results[key] = ges_result
            print(f'ges: {ges_result}')

    def test_pc(self,result,ground_truth):
        obj = PC()
        glasso = cdt.independence.graph.Glasso()
        skeleton = glasso.predict(result)
        output = obj.predict(skeleton)
        output = nx.to_numpy_array(output)
        SHD_result = SHD(ground_truth.to_numpy(), output)
        return SHD_result

    def test_ges(self,result,ground_truth):
        obj = GES()
        pred = obj.create_graph_from_data(result)
        pred = nx.to_numpy_array(pred)
        SHD_result = SHD(ground_truth.to_numpy(), pred)
        return SHD_result

    def save_results(self):
        timestamp = datetime.datetime.now()
        timestamp = str(timestamp.strftime("%Y%m%d%H%M%S"))
        np.savetxt(f'results/results_{timestamp}.csv', self.test_results, delimiter=',', fmt='%d')

    def save_test_results(self):
        timestamp = datetime.datetime.now()
        timestamp = str(timestamp.strftime("%Y%m%d%H%M%S"))
        np.savetxt(f'results/test_results_{timestamp}.csv', self.test_results, delimiter=',', fmt='%d')

    def load_generator(self,abs_path): 
        path = abs_path.split('/')[-1] #takes the ABSOLUTE PATH
        matches = re.findall(r'\d+\.\d+|\d+', path)
        latent_dim = int(matches[0])
        hidden_dim = int(matches[1])
        lr = float(matches[2])
        epochs = int(matches[3])
        c = CIGAN7.CIGAN(self.dataset, self.batch_size, latent_dim, hidden_dim, lr, epochs, sample_interval)
        gen = c.get_generator()
        gen.load_state_dict(torch.load(abs_path))
        print(gen)
        c.set_generator(gen)
        self.model = c

    def load_discriminator(self,path):
        path = path.split('/')[-1] #takes the ABSOLUTE PATH
        matches = re.findall(r'\d+\.\d+|\d+', path)
        latent_dim = int(matches[0])
        hidden_dim = int(matches[1])
        lr = float(matches[2])
        epochs = int(matches[3])
        c = CIGAN7.CIGAN(self.dataset, self.batch_size, latent_dim, hidden_dim, lr, epochs, sample_interval)
        gen = c.get_discriminator()
        gen.load_state_dict(torch.load(path))
        c.set_discriminator(gen)
        self.model = c

# t = Tester()
# t.set_test_range(latent_dim=(1,500),hidden_dim=(5,300),lr=(0.0002,0.0009),epochs=(50,200))
# t.set_dataset('datasets/sachs_data_noindex.csv')
# t.run()
# t.save_results()
# t.test()
# t.save_test_results()
# print('done')