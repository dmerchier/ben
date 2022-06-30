import sys

from utils.nn import create_nn

if __name__ == '__main__':
    bin_dir = sys.argv[1]  # binary files generated for the training
    out_dir = sys.argv[2]  # folder path where create model files

    create_nn(bin_dir, out_dir, 'dummy')