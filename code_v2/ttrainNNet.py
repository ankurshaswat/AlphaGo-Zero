import os
import pickle

from Compete import compete
from NNet import NNet

args = {}

if __name__ == "__main__":
    new_example_dir = './new_examples/'
    used_example_dir = './used_examples/'

    new_net = NNet(path='model.pytorch')

    all_examples = []

    for example_file in os.listdir(new_example_dir):
        with open(new_example_dir + example_file, 'rb') as handle:
            b = pickle.load(handle)
            all_examples += b['examples']

    new_net.train(all_examples)
    old_net = NNet(path='model.pytorch')

    oldNN_win_count, newNN_win_count = compete(
        old_NN=old_net, new_NN=new_net, args=args)

    if newNN_win_count > oldNN_win_count:
        new_net.save_checkpoint(path='model.pytorch')
        new_net.save_checkpoint(path='best_model.pytorch')
        for example_file in os.listdir(new_example_dir):
            os.rename(new_example_dir + example_file,
                      used_example_dir + example_file)