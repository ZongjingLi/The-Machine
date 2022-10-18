import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name",default = "Violet Evergarden",type = str, help = "the Violet Evergarden")

config = parser.parse_args(args = [])