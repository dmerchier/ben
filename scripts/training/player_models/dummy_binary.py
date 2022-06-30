import sys

from utils.binary import execute_binary_script

if __name__ == '__main__':
    train_file = sys.argv[1]  # file path for training
    valid_file = sys.argv[2]  # file path for validating
    out_dir = sys.argv[3]  # folder path where create binary files

    execute_binary_script(train_file, valid_file, out_dir, 2)
