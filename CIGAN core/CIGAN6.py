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

#-=-= hyperparameters =-=-=
batch_size = 50
latent_dim = 100
hidden_dim = 200
lr=0.0002 #0.0002
epochs = 200
sample_interval = 400
#-=-=-=-= data =-=-=-=-=

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
    
# dataset = CSVDataset('datasets/sachs_data_noindex.csv')
# dataset = CSVDataset('datasets/UF_sorted.csv') #this one doesnt work for some reason
# dataset = CSVDataset('datasets/river_runoff.csv')

# train_dev_sets = ConcatDataset([sachs_dataset, uf_dataset, river_dataset])
# train_dev_loader = DataLoader(dataset=train_dev_sets, batch_size=batch_size, shuffle=True)
train_dev_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) #cant use multiple datasets because they are different size

#-=-=-=-= model =-=-=-=-=

def generate_adj_A(data_dim):
    nodes = data_dim
    graph = nx.erdos_renyi_graph(nodes, 0.25, directed=True) #erdos renyi graph
    adj_mat_np = nx.adjacency_matrix(graph).todense()
    adj_mat = torch.from_numpy(adj_mat_np).to(torch.float32)
    #save adj_mat to csv
    adj_mat_np = adj_mat_np.astype(int)
    timestamp = datetime.datetime.now()
    timestamp = str(timestamp.strftime("%Y%m%d%H%M%S"))
    np.savetxt(f'adj_mat_{timestamp}.csv', adj_mat_np, delimiter=',', fmt='%d')
    return adj_mat

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()

        self.adj_A = generate_adj_A(output_dim)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Identity()
        )

        self.resize_to_adj_size = nn.Linear(input_dim, output_dim)
        self.resize_x_back = nn.Linear(output_dim, input_dim)

    def forward(self, x):
        
        resize_x_for_adj = self.resize_to_adj_size(x)
        x2 = torch.matmul(self.adj_A, resize_x_for_adj.T) 
        x = self.resize_x_back(x2.T)
        x = self.encoder(x)
        # logits = torch.matmul(self.adj_A, x)
        x = self.decoder(x)
        return x
    
    def get_adj_A(self):
        return self.adj_A


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        # Discriminator layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

#-=-=- init stuff -=-=-=
gan_output_dim = dataset[0].shape[0]

generator = Generator(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=gan_output_dim) ##############
discriminator = Discriminator(input_dim=gan_output_dim, hidden_dim=hidden_dim, output_dim=1)

generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()

#-=-=-=-= training =-=-=-=-=

for epoch in range(epochs):
    for idx, real_data in enumerate(train_dev_loader):

        noise = torch.randn(batch_size,latent_dim)
        fake_data = generator(noise)
        #- Train Discriminator

        discriminator_optimizer.zero_grad()

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data.detach())
        real_loss = loss_fn(real_output, real_labels)
        fake_loss = loss_fn(fake_output, fake_labels)
        discriminator_loss = real_loss + fake_loss

        discriminator_loss.backward()
        discriminator_optimizer.step()

        #- Train Generator

        generator_optimizer.zero_grad()

        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)

        fake_labels = torch.ones(batch_size, 1)
        fake_output = discriminator(fake_data)
        generator_loss = loss_fn(fake_output, fake_labels) 
        # !!! could do ELBO loss here?????????
        generator_loss.backward()
        generator_optimizer.step()

        #-=- output info and save
        batches_done = epoch * len(train_dev_loader) + idx
        if batches_done % sample_interval == 0:
                # save_image(fake_data.data, "images/%d.png" % batches_done, normalize=True)
                timestamp = datetime.datetime.now()
                timestamp = str(timestamp.strftime("%Y%m%d%H%M%S"))
                df = pd.DataFrame(fake_data.detach().numpy(), columns=dataset.get_col_names())
                df.to_csv((f'images/%d_{timestamp}.csv' % batches_done), index=False)
                print("saved a table!")
    print("Epoch %d: Generator loss=%.4f, Discriminator loss=%.4f" % (epoch+1, generator_loss.item(), discriminator_loss.item()))

