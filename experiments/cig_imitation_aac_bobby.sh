#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume]"
    exit 1
fi

COMMAND="$1"
CHECK_POINT=$BASEDIR/checkpoints/cig_imitation_aac_bobby_cp.pth
H5_PATH=~/test/datasets/vizdoom/bobby_cig_map01/flat.h5

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/imitation.py \
    --h5_path $H5_PATH \
    --batch_size 100 \
    --skiprate 1 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 1
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/imitation.py \
    --h5_path $H5_PATH \
    --batch_size 100 \
    --load $CHECK_POINT \
    --skiprate 1 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 1
else
    echo "'$COMMAND' is unknown command."
fi