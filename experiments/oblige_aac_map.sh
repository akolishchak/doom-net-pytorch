#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=aac_map
CHECK_POINT=$BASEDIR/checkpoints/oblige_aac_map_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_test.npy
CONFIG=$BASEDIR/environments/oblige-map.cfg
INSTANCE=oblige


if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 100 \
    --batch_size 20 \
    --episode_discount 0.99 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100
    --episode_num 20000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 20 \
    --batch_size 20 \
    --episode_discount 0.95 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 1 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi