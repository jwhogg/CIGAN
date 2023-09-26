from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import pandas as pd
import argparse
import sklearn.preprocessing as skp

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("filename",type=str)
    parser.add_argument("-sample_size",type=int,default=10)
    args=parser.parse_args()
    return args

args=parse_args()
fn = args.filename
sample_size = args.sample_size

# read in the .bif file
reader = BIFReader(fn)

# access the Bayesian Network object
bn = reader.get_model()

# create a sampler object
sampler = BayesianModelSampling(bn)

# generate 10 samples from the network
samples = sampler.forward_sample(size=sample_size)

# convert the samples to a pandas DataFrame
df = pd.DataFrame(samples)

# print(df)

df_np = df.to_numpy()

enc = skp.OrdinalEncoder()

enc.fit(df)
mat_encoded = enc.transform(df_np)
df_encoded = pd.DataFrame(mat_encoded, columns = list(df.columns.values))

# print(df_encoded)

# write the DataFrame to a CSV file
df_encoded.to_csv(f'{fn[0:-4]}.csv', index=False)
print("saved!")


