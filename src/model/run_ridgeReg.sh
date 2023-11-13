#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=deepgaze
#SBATCH --output=/project/rrg-pbellec/mstlaure/friends_algonauts/slurm_files/slurm-%A_%a.out
#SBATCH --error=/project/rrg-pbellec/mstlaure/friends_algonauts/slurm_files/slurm-%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marie.stl@gmail.com

# load modules
module load python/3.8.2

# activate project's virtual env
source /home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/fa_venv/bin/activate

INPUT="/home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/processed_data/RR_features"
BIDSDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/data/friends.timeseries/timeseries"
ATLASDIR="/home/mstlaure/projects/rrg-pbellec/mstlaure/friends_algonauts/data/friends.timeseries/subject_masks/tpl-MNI152NLin2009cAsym"
OUTPUT="/project/rrg-pbellec/mstlaure/friends_algonauts/results/RR_scores"
SUBJECT_NUM="${1}" # 01, 02, 03

# launch job
python -m model_ridgereg \
        --idir ${INPUT} \
        --bdir ${BIDSDIR} \
        --odir ${OUTPUT} \
        --adir ${ATLASDIR} \
        --participant "sub-${SUBJECT_NUM}" \
        --modalities visual audio \
        --back 5 \
        --input_duration 3 \
        --n_split 10 \
        --random_state 7 \
        --verbose 1
