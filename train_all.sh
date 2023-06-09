#!/bin/bash

# Default epochs value
EPOCHS=${1:-1}

# Declare a list of already completed tasks
declare -A completed_tasks=(
    # ["action_classification-multilabel-True-True-False"]=1
    # ["action_classification-multilabel-True-True-True"]=1
    # ["action_classification-multilabel-True-True-True"]=1
    # ["action_classification-multilabel-True-True-False"]=1
    # ["action_classification-multilabel-True-False-False"]=1
    # ["action_classification-multilabel-False-True-False"]=1
    # ["action_classification-multilabel-False-True-True"]=1
    # ["action_classification-multilabel-False-False-True"]=1
    # ["action_classification-multilabel-True-False-True"]=1
    # ["segmentation-deeplabv3-True-True-True"]=1
    # ["segmentation-deeplabv3-True-True-False"]=1
    # ["segmentation-deeplabv3-True-False-False"]=1
    # ["segmentation-deeplabv3-False-True-False"]=1
    # ["segmentation-deeplabv3-False-True-True"]=1
    # ["segmentation-deeplabv3-False-False-True"]=1
    # ["segmentation-deeplabv3-True-False-True"]=1
    # ["segmentation-deeplabv3-True-True-True"]=1
    # ["segmentation-deeplabv3-True-True-False"]=1
    # ["segmentation-deeplabv3-True-False-False"]=1
    # ["segmentation-deeplabv3-True-False-True"]=1
    # ["segmentation-unet-True-True-True"]=1
    # ["segmentation-unet-True-True-False"]=1
    # ["segmentation-unet-True-False-False"]=1
    # ["segmentation-unet-False-True-False"]=1
    # ["segmentation-unet-False-True-True"]=1
    # ["segmentation-unet-False-False-True"]=1
    # ["segmentation-unet-True-False-True"]=1
)


# Loop over tasks
for TASK in action_classification segmentation; do

    # Loop over architectures
    for ARCH in dependent unet deeplabv3; do

        # Skip invalid task-architecture combinations
        if [ "$TASK" = "segmentation" ] && ([ "$ARCH" = "multilabel" ] || [ "$ARCH" = "dependent" ]); then
            continue
        fi
        if [ "$TASK" = "action_classification" ] && ([ "$ARCH" = "unet" ] || [ "$ARCH" = "deeplabv3" ]); then
            continue
        fi

        # Loop over all combinations of RGB, depth, and thermal input
        for RGB in "--rgb" "--no-rgb"; do
            for DEPTH in "--depth" "--no-depth"; do
                for THERMAL in "--thermal" "--no-thermal"; do

                    # If all input types are False, skip this iteration
                    if [ "$RGB" = "--no-rgb" ] && [ "$DEPTH" = "--no-depth" ] && [ "$THERMAL" = "--no-thermal" ]; then
                        continue
                    fi

                    key="$TASK-$ARCH-$( [[ $RGB == *"--no"* ]] && echo "False" || echo "True")-$( [[ $DEPTH == *"--no"* ]] && echo "False" || echo "True")-$( [[ $THERMAL == *"--no"* ]] && echo "False" || echo "True")"
                    
                    echo "checking key $key"
                    # If this task has already been completed, skip this iteration
                    if [ ${completed_tasks[$key]+isset} ]; then
                        continue
                    fi

                    echo "Running with task=$TASK, architecture=$ARCH, rgb=$RGB, depth=$DEPTH, thermal=$THERMAL, epochs=$EPOCHS"
                    python -m tristar.train --task "$TASK" --architecture "$ARCH" $RGB $DEPTH $THERMAL --epochs "$EPOCHS"
                done
            done
        done
    done
done
