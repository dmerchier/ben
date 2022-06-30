import sys

from utils.nn import continue_nn

if __name__ == '__main__':
    bin_dir = sys.argv[1]  # binary files generated for the training
    checkpoint_model = sys.argv[2]  # pretrained model e.g './models/lefty-1000000'
    out_dir = sys.argv[3]  # folder path where create model files

    continue_nn(bin_dir, checkpoint_model, out_dir, 'lefty')
