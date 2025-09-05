#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=sam2_scratch
#
# Specify output file.
#SBATCH --output=sam2_%j.log
#
# Specify error file.
#SBATCH --error=sam2_%j.err
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

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

# Load modules.
module load stack/2024-06 python/3.11 cuda/12.4
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for sam2.
source /cluster/scratch/niacobone/sam2/myenv/bin/activate
echo "Activated Python venv: $(which python)"

# Execute

exclude_list=("video1" "video2" "video3")  # Add video names (without extension) to exclude

cd /cluster/scratch/niacobone/sam2
echo "Starting sam2 inference..."
dir_name='helloworld'
echo "Processing video: $dir_name"
python inference.py --input_dir="$dir_name"
echo "----------------------------------"

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"
