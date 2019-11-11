#!/bin/bash

CUDA_DEVICES=(0 1 2 3)
NUM_THREADS=28
NUM_CYCLES=5

NUM_GPU=${#CUDA_DEVICES[@]}
echo "Num gpus available = $NUM_GPU"

kill_child_processes() {
    isTopmost=$1
    curPid=$2
    childPids=$(ps -o pid --no-headers --ppid ${curPid})
    for childPid in $childPids; do
        kill_child_processes 0 $childPid
    done
    if [ $isTopmost -eq 0 ]; then
        kill -9 $curPid 2>/dev/null
    fi
}

# Ctrl-C trap. Catches INT signal
trap "kill_child_processes 1 $$; exit 0" INT

mkdir ../logs
unique_token="$(date +"%T")"

# python create_new_net.py


CYCLE_NUM=0
while [[ $CYCLE_NUM -lt $NUM_CYCLES ]]; do


    # echo "Spawning threads for self play."

    # THREAD_NUM=0
    # CUDA_DEVICE=0
    # while [[ $THREAD_NUM -lt $NUM_THREADS ]]; do
    #     # echo CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$CUDA_DEVICE]} CUDA_DEVICE_VAR=$CUDA_DEVICE THREAD_NUM=$THREAD_NUM UNIQUE_TOKEN=$unique_token CYCLE_NUM=$CYCLE_NUM
    #     CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[CUDA_DEVICE]} python game_generator.py -thread_num $THREAD_NUM -unique_token $unique_token >> ../logs/${unique_token}_${THREAD_NUM}.log &
    #     ((THREAD_NUM = THREAD_NUM + 1))
    #     ((CUDA_DEVICE = (CUDA_DEVICE + 1) % NUM_GPU))
    # done

    # wait

    # echo "Starting Training."
    # CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python train_net.py >> ../logs/${unique_token}.log

    # echo "Spawning threads for competing."

    THREAD_NUM=0
    CUDA_DEVICE=0
    while [[ $THREAD_NUM -lt $NUM_THREADS ]]; do
        # echo CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$CUDA_DEVICE]} CUDA_DEVICE_VAR=$CUDA_DEVICE THREAD_NUM=$THREAD_NUM UNIQUE_TOKEN=$unique_token CYCLE_NUM=$CYCLE_NUM
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[CUDA_DEVICE]} python compete_with_best.py -thread_num $THREAD_NUM >> ../logs/${unique_token}_${THREAD_NUM}.log &
        ((THREAD_NUM = THREAD_NUM + 1))
        ((CUDA_DEVICE = (CUDA_DEVICE + 1) % NUM_GPU))
    done

    wait

    echo "Starting result compilation."
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[0]} python compile_results.py >> ../logs/${unique_token}.log

    #rm -rf ../compete_results

    ((CYCLE_NUM = CYCLE_NUM + 1))
done
