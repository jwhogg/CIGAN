import os

with open("datasets_to_sample.txt","r") as f:
	lines = f.readlines()

for path in lines:
	os.system(f'python3.9 bayesian_sampler.py "{path[0:-1]}" -sample_size=2500')