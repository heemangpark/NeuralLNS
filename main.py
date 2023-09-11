import argparse

from src.destroy.core import run

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
args = parser.parse_args()

if __name__ == '__main__':
    run(args.mode)
