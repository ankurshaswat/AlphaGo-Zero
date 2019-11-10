#!/bin/bash

NUM_THREADS=10


kill_child_processes() {
    isTopmost=$1
    curPid=$2
    childPids=`ps -o pid --no-headers --ppid ${curPid}`
    for childPid in $childPids
    do
        kill_child_processes 0 $childPid
    done
    if [ $isTopmost -eq 0 ]; then
        kill -9 $curPid 2> /dev/null
    fi
}

# Ctrl-C trap. Catches INT signal
trap "kill_child_processes 1 $$; exit 0" INT


# python create_new_net.py


for i in {0..1}
do

    unique_token="$(date +"%T")"

    echo "Spawning new threads with unique token $unique_token"

    i=1
    while [[ $i -le $NUM_THREADS ]]
    do
        python game_generator.py -thread_num $i -unique_token $unique_token &
        ((i = i + 1))
    done

    ## Wait for games to finish
    
    wait
    echo "All threads of this batch complete."

    python train_net.py

    i=1
    while [[ $i -le $NUM_THREADS ]]
    do
        python compete_with_best.py -thread_num $i &
        ((i = i + 1))
    done

    python compile_results.py

    rm -rf ../compete_results
    
done

