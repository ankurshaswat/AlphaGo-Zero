#!/bin/bash

# python createNewNet.py

for i in {0..10}
do

    NUM_THREADS=5
    unique_token="$(date +"%T")"


    i=1
    while [[ $i -le $NUM_THREADS ]]
    do
        python gameGenerator.py $init_net $i $unique_token &
        ((i = i + 1))
    done

    ## Wait for games to finish
    wait

    # python trainNNet.py
done

