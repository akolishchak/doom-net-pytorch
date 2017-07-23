#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume]"
    exit 1
fi

COMMAND="$1"
CHECK_POINT=$BASEDIR/checkpoints/cig_imitation_aac_frames4_cp.pth
H5_PATH=~/test/datasets/vizdoom/cig_map01/flat.h5

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/imitation_frames.py \
    --h5_path $H5_PATH \
    --batch_size 100 \
    --skiprate 1 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 5000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/imitation_frames.py \
    --h5_path $H5_PATH \
    --batch_size 100 \
    --load $CHECK_POINT \
    --skiprate 1 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 5000
else
    echo "'$COMMAND' is unknown command."
fi