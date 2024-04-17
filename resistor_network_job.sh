#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.


#SBATCH -J resistor_network_DNN #Slurm job name

# Set the maximum runtime, uncomment if you need it
#SBATCH -t 1:00:00 #Maximum runtime of 1 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
#SBATCH --mail-user=chchengap@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) 
#SBATCH -p gpu-share

# To use 1 cpu cores and 1 gpu devices in a node
#SBATCH -N 1 -n 1 --gres=gpu:2

# Setup runtime environment if necessary
module load cuda
module load anaconda3/2021.05
source activate my_env



# Go to the job submission directory and run your application
cd $HOME/apps/resistor_network_HPC/src
$HOME/.conda/envs/my_env/bin/python ./main.py