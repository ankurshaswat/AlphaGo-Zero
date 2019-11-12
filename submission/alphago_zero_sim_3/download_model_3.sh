#!/bin/bash

# TO CHECK THIS ON A LINUX SYSTEM / SPECIFICALLY HPC

# https://drive.google.com/file/d/1vCsWjyCrBM7xo1pjUo6ttL_tD2SnzdyN/view?usp=sharing
# https://drive.google.com/file/d/1vCsWjyCrBM7xo1pjUo6ttL_tD2SnzdyN/view?usp=sharing
# wget -O file https://googledrive.com/host/[ID]

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1vCsWjyCrBM7xo1pjUo6ttL_tD2SnzdyN' -O model.pytorch3
mkdir model_3
mv model.pytorch3 model_3/model.pytorch3
