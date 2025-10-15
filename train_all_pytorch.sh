#!/bin/bash

# Array to track completed tasks
declare -A completed_tasks

# Function to run training
run_training() {
    local task=$1
    local architecture=$2
    local rgb=$3
    local depth=$4
    local thermal=$5
    local epochs=${6:-10}
    local batch_size=${7:-8}

    # Create a unique identifier for this task
    local task_id="${task}-${architecture}-${rgb}-${depth}-${thermal}"

    # Check if task is already completed
    if [ "${completed_tasks[$task_id]}" == "1" ]; then
        echo "Skipping already completed task: $task_id"
        return 0
    fi

    echo "=================================="
    echo "Starting training: $task_id"
    echo "=================================="

    # Build the command
    local cmd="python -m tristar.train_pytorch --task $task --architecture $architecture --epochs $epochs --batch_size $batch_size"

    # Add modality flags
    if [ "$rgb" == "true" ]; then
        cmd="$cmd --rgb"
    else
        cmd="$cmd --no-rgb"
    fi

    if [ "$depth" == "true" ]; then
        cmd="$cmd --depth"
    else
        cmd="$cmd --no-depth"
    fi

    if [ "$thermal" == "true" ]; then
        cmd="$cmd --thermal"
    else
        cmd="$cmd --no-thermal"
    fi

    # Run the training
    echo "Running: $cmd"
    eval $cmd

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $task_id"
        completed_tasks[$task_id]="1"
    else
        echo "✗ Failed: $task_id"
        return 1
    fi
}

# Set default parameters
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}

echo "=================================="
echo "Batch Training Script (Pure PyTorch)"
echo "=================================="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=================================="

# Action Classification - Dependent Architecture
# RGB + Depth + Thermal
run_training "action_classification" "dependent" "true" "true" "true" $EPOCHS $BATCH_SIZE

# RGB + Depth
run_training "action_classification" "dependent" "true" "true" "false" $EPOCHS $BATCH_SIZE

# RGB + Thermal
run_training "action_classification" "dependent" "true" "false" "true" $EPOCHS $BATCH_SIZE

# Depth + Thermal
run_training "action_classification" "dependent" "false" "true" "true" $EPOCHS $BATCH_SIZE

# RGB only
run_training "action_classification" "dependent" "true" "false" "false" $EPOCHS $BATCH_SIZE

# Depth only
run_training "action_classification" "dependent" "false" "true" "false" $EPOCHS $BATCH_SIZE

# Thermal only
run_training "action_classification" "dependent" "false" "false" "true" $EPOCHS $BATCH_SIZE

# Action Classification - MultiLabel Architecture
# RGB + Depth + Thermal
run_training "action_classification" "multilabel" "true" "true" "true" $EPOCHS $BATCH_SIZE

# RGB + Depth
run_training "action_classification" "multilabel" "true" "true" "false" $EPOCHS $BATCH_SIZE

# RGB + Thermal
run_training "action_classification" "multilabel" "true" "false" "true" $EPOCHS $BATCH_SIZE

# Depth + Thermal
run_training "action_classification" "multilabel" "false" "true" "true" $EPOCHS $BATCH_SIZE

# RGB only
run_training "action_classification" "multilabel" "true" "false" "false" $EPOCHS $BATCH_SIZE

# Depth only
run_training "action_classification" "multilabel" "false" "true" "false" $EPOCHS $BATCH_SIZE

# Thermal only
run_training "action_classification" "multilabel" "false" "false" "true" $EPOCHS $BATCH_SIZE

# Human Segmentation - UNet Architecture
# RGB + Depth + Thermal
run_training "segmentation" "unet" "true" "true" "true" $EPOCHS $BATCH_SIZE

# RGB + Depth
run_training "segmentation" "unet" "true" "true" "false" $EPOCHS $BATCH_SIZE

# RGB + Thermal
run_training "segmentation" "unet" "true" "false" "true" $EPOCHS $BATCH_SIZE

# Depth + Thermal
run_training "segmentation" "unet" "false" "true" "true" $EPOCHS $BATCH_SIZE

# RGB only
run_training "segmentation" "unet" "true" "false" "false" $EPOCHS $BATCH_SIZE

# Depth only
run_training "segmentation" "unet" "false" "true" "false" $EPOCHS $BATCH_SIZE

# Thermal only
run_training "segmentation" "unet" "false" "false" "true" $EPOCHS $BATCH_SIZE

# Human Segmentation - DeepLabV3 Architecture
# RGB + Depth + Thermal
run_training "segmentation" "deeplabv3" "true" "true" "true" $EPOCHS $BATCH_SIZE

# RGB + Depth
run_training "segmentation" "deeplabv3" "true" "true" "false" $EPOCHS $BATCH_SIZE

# RGB + Thermal
run_training "segmentation" "deeplabv3" "true" "false" "true" $EPOCHS $BATCH_SIZE

# Depth + Thermal
run_training "segmentation" "deeplabv3" "false" "true" "true" $EPOCHS $BATCH_SIZE

# RGB only
run_training "segmentation" "deeplabv3" "true" "false" "false" $EPOCHS $BATCH_SIZE

# Depth only
run_training "segmentation" "deeplabv3" "false" "true" "false" $EPOCHS $BATCH_SIZE

# Thermal only
run_training "segmentation" "deeplabv3" "false" "false" "true" $EPOCHS $BATCH_SIZE

echo "=================================="
echo "All training tasks completed!"
echo "=================================="
echo "Summary of completed tasks:"
for task_id in "${!completed_tasks[@]}"; do
    echo "  ✓ $task_id"
done
