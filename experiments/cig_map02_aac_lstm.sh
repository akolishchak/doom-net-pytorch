#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=aac_lstm
CHECK_POINT=$BASEDIR/checkpoints/cig_map02_aac_lstm_30_97_skip4_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_cig2_speed.npy
CONFIG=$BASEDIR/environments/cig2.cfg

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 30 \
    --batch_size 20 \
    --episode_discount 0.97 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --base_model $BASEDIR/trained_models/cig_map02_imitation_model_aac_lstm.pth \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 30 \
    --batch_size 20 \
    --episode_discount 0.97 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 500
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi