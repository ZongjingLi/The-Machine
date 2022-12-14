import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name",   default = "Violet Evergarden",type = str, help = "the Violet Evergarden")
# specification of the language to program parser
parser.add_argument("--corpus", default = "assets/corpus.txt"  ,type = str)
parser.add_argument("--grammar",default = "assets/grammar.json",type = str)
parser.add_argument("--word_dim",  default = 128,type = int)
parser.add_argument("--latent_dim",default = 132,type = int)
parser.add_argument("--concept_dim",default = 100,type = int)
parser.add_argument("--concepts",   default = None)
config = parser.parse_args(args = [])