#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume]"
    exit 1
fi

COMMAND="$1"
CHECK_POINT=$BASEDIR/checkpoints/cig_map02_imitation_aac_lstm_cp.pth
H5_PATH=~/test/datasets/vizdoom/cig_map02/flat.h5

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/imitation_lstm.py \
    --h5_path $H5_PATH \
    --episode_size 20 \
    --batch_size 64 \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/imitation_lstm.py \
    --h5_path $H5_PATH \
    --episode_size 20 \
    --batch_size 64 \
    --load $CHECK_POINT \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500
else
    echo "'$COMMAND' is unknown command."
fi