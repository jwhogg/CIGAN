# CIGAN
University dissertation research project- Causal Implicit GAN, Causal Discovery using a novel GAN structure

## Quickstart

- It is reccomended that you use a jupyter notebook to get started, you can set up the model like this:
  `temp_model = CIGAN7.CIGAN(dataset,batch_size,latent_dim,hidden_dim,lr,epochs,sample_interval)`

- See the jupyter notebook 'testing.ipynb' for an example on how to run a dataset on the model

- Some dataset files are included in this repo, but the model must take the data as a numpy array, with no headers, examples on how the data is converted can be found in the 'other' folder

- Code to visualise the results is availible, and the model can also output images of the results

- A review of the model on different types of data can be found in an excel file in 'other'



- Some code and design in based on DAG-WGAN (https://arxiv.org/abs/2204.00387)
- Please reference my paper and DAG-WGAN if any code is used.
