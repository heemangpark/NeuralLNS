import argparse

from src.destroy.core import run

parser = argparse.ArgumentParser(description='RUNNING MAIN CODE (DEFAULT = TRAIN)')
parser.add_argument('--mode', choices=['train', 'eval', 'train_data', 'eval_data'], default='train')
args = parser.parse_args()

if __name__ == '__main__':
    run(args.mode)
