import sys
import pickle
import os
from Generator import generateEpisodes
from NNet import NNet

args = {

}

if __name__ == "__main__":
    new_example_dir = './new_examples/'

    if not os.path.exists(new_example_dir):
        os.makedirs(new_example_dir)

    init_net = sys.argv[1]
    thread_num = int(sys.argv[2])
    unique_name = sys.argv[3]

    net = NNet(path='model.pytorch')

    epis = generateEpisodes(net, args)

    # save epis as pickle using uniqueName_thread
    to_save = {'examples': epis}

    with open(new_example_dir + unique_name+'_'+str(thread_num)+'.pickle', 'wb') as handle:
        pickle.dump(to_save, handle)
