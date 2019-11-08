#!/bin/bash

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


# python createNewNet.py

for i in {0..1}
do

    NUM_THREADS=10
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
done

