import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default   = "Violet Evergarden",type = str, help = "the Violet Evergarden")
parser.add_argument("--corpus",default  = "data/corpus.txt"  ,type = str)
parser.add_argument("--grammar",default = "data/grammar.json",type = str)
config = parser.parse_args(args = [])