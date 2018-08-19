#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=ppo_screen
CHECK_POINT=$BASEDIR/checkpoints/oblige_ppo_screen_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_test_forward.npy
CONFIG=$BASEDIR/environments/oblige/oblige-map.cfg
INSTANCE=oblige


if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 150 \
    --batch_size 110 \
    --episode_discount 0.99333333333 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 50 \
    --episode_num 50000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 150 \
    --batch_size 110 \
    --episode_discount 0.99333333333 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 10 \
    --episode_num 50000
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --batch_size 1 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi