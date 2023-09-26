#helper file to change dataset without having to go into config.py to do it

#usage:
#  eg: python3.9 choose_dataset.py auto-mpg.csv

import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("filename",type=str)
    args=parser.parse_args()
    return args

args=parse_args()
fn = args.filename

with open('file_pointer.txt','w') as f:
	f.write(fn)

