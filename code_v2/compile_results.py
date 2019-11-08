"""
Compile Results and get best model
"""

import os

from utils import parse_args

if __name__ == "__main__":
    ARGS = parse_args()

    OLD_WIN_COUNT, NEW_WIN_COUNT = 0, 0

    for example_file in os.listdir('../compete_results'):
        with open('../compete_results/' + example_file, 'r') as handle:
            data = handle.read()
            scores = data.split()
            OLD_WIN_COUNT += int(scores[0])
            NEW_WIN_COUNT += int(scores[1])

    if NEW_WIN_COUNT > OLD_WIN_COUNT:
        os.rename(ARGS.temp_model_path+str(ARGS.type),
                  ARGS.best_model_path+str(ARGS.type))
