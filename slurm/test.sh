#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=test_euler
#
# Specify output file.
#SBATCH --output=test_euler_%j.log
#
# Specify error file.
#SBATCH --error=test_euler_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=00:05:00
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=1
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090:1

module load stack/2024-06 python/3.11 cuda/12.4
source /cluster/scratch/niacobone/sam2/myenv/bin/activate
cd /cluster/scratch/niacobone/sam2
echo "Starting test..."

python -u test.py