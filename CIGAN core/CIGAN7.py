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
# batch_size = 50
# latent_dim = 100
# hidden_dim = 200
# lr=0.0002 #0.0002
# epochs = 200
# sample_interval = 400
#-=-=-=-= data =-=-=-=-=

class CIGAN:
    def __init__(self, dataset, batch_size, latent_dim, hidden_dim, lr, epochs, sample_interval): #dataset should be dataloader
        self.dataset = dataset
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.sample_interval = sample_interval
        self.gan_output_dim = self.dataset[0].shape[0]

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        self.debug = False

        self.generator = Generator(input_dim=self.latent_dim, hidden_dim=self.hidden_dim, output_dim=self.gan_output_dim)
        self.discriminator = Discriminator(input_dim=self.gan_output_dim, hidden_dim=self.hidden_dim, output_dim=1)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.loss_fn = nn.BCELoss()

        self.G_losses = []
        self.D_losses = []
    
    def train(self):
        for epoch in range(self.epochs):
            for idx, real_data in enumerate(self.dataloader):

                noise = torch.randn(self.batch_size,self.latent_dim)
                fake_data = self.generator(noise)
                #- Train Discriminator

                self.discriminator_optimizer.zero_grad()

                real_labels = torch.ones(self.batch_size, 1)
                fake_labels = torch.zeros(self.batch_size, 1)
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data.detach())
                real_loss = self.loss_fn(real_output, real_labels)
                fake_loss = self.loss_fn(fake_output, fake_labels)
                discriminator_loss = real_loss + fake_loss

                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                #- Train Generator

                self.generator_optimizer.zero_grad()

                noise = torch.randn(self.batch_size, self.latent_dim)
                fake_data = self.generator(noise)

                fake_labels = torch.ones(self.batch_size, 1)
                fake_output = self.discriminator(fake_data)
                generator_loss = self.loss_fn(fake_output, fake_labels) 
                generator_loss.backward()
                self.generator_optimizer.step()

                self.G_losses.append(generator_loss.item())
                self.D_losses.append(discriminator_loss.item())

            if self.debug:
                print("Epoch %d: Generator loss=%.4f, Discriminator loss=%.4f" % (epoch+1, generator_loss.item(), discriminator_loss.item()))
        
    def set_generator(self, arg):
        if arg != None:
            self.generator = arg
            self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
    
    def set_discriminator(self,arg):
        if arg != None:
            self.discriminator = arg
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def get_synthetic_sample(self):
        noise = torch.randn(self.batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        synthetic_sample = pd.DataFrame(fake_data.detach().numpy(), columns=self.dataset.get_col_names())
        return synthetic_sample

    def get_G_losses(self):
        return self.G_losses

    def get_D_losses(self):
        return self.D_losses

    def get_adj_A(self):
        return self.generator.get_adj_A()

    def set_debug(self, debug):
        self.debug = debug

    def get_generator(self):
        return self.generator
    
    def get_discriminator(self):
        return self.discriminator
        
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()

        self.adj_A = self.generate_adj_A(output_dim)

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

    def generate_adj_A(self, data_dim):
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

