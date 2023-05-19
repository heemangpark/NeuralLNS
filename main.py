import argparse

from src.destroy.core import run

parser = argparse.ArgumentParser(description='RUNNING MAIN CODE !')
parser.add_argument('--mode', choices=['train', 'eval'], default='train')
args = parser.parse_args()

if __name__ == '__main__':
    run(args.mode)
