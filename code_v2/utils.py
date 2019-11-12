"""
Util functions
"""
import argparse


def parse_args():
    """
    Argument Parser.
    """
    parser = argparse.ArgumentParser(description='Generate New Games.')

    #parser.add_argument('-cuda', action='store_true', default=False)
    parser.add_argument('-cuda', action='store_true', default=True)

    parser.add_argument('-new_path', action='store',
                        dest="new_path", default='../new_examples/')
    parser.add_argument('-used_path', action='store',
                        dest="used_path", default='../used_examples/')
    parser.add_argument('-thread_num', action='store',
                        dest="thread_num", type=int, default=0)
    parser.add_argument('-unique_token', action='store',
                        dest="unique_token", default='notSpecified')
    parser.add_argument('-numEpisodes', action='store',
                        dest='numEpisodes', type=int, default=5)#10
    parser.add_argument('-numSimulations', action='store',
                        dest='numSimulations', type=int, default=100)#100
    parser.add_argument('-cpuct', action='store',
                        dest='cpuct', type=float, default=1.0)
    parser.add_argument('-numStepsForTempChange', action='store',
                        dest='numStepsForTempChange', type=int, default=30)
    parser.add_argument('-netType', action='store',
                        dest='type', type=int, default=3)
    parser.add_argument('-history', action='store_true',
                        dest='history', default=False)
    parser.add_argument('-dropout', action='store',
                        dest='dropout', type=float, default=0.2)
    parser.add_argument('-lr', action='store',
                        dest='lr', type=float, default=0.001)
    parser.add_argument('-momentum', action='store',
                        dest='momentum', type=float, default=0.9)
    parser.add_argument('-l2_regularization', action='store',
                        dest='l2_regularization', type=float, default=1e-5)#1e-5
    parser.add_argument('-epochs', action='store',
                        dest='epochs', type=int, default=10)#10
    parser.add_argument('-batch_size', action='store',
                        dest='batch_size', type=int, default=32)
    parser.add_argument('-numGamesPerSide', action='store',
                        dest='numGamesPerSide', type=int, default=2)#10
    parser.add_argument('-best_model_path', action='store',
                        dest='best_model_path', default='../models/best_model.pytorch')
    parser.add_argument('-temp_model_path', action='store',
                        dest='temp_model_path', default='../models/temp_model.pytorch')

    args = parser.parse_args()

    return args
