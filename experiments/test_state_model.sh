#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=state
STATE_MODEL=$BASEDIR/checkpoints/state_model_cp.pth
CHECK_POINT=$BASEDIR/checkpoints/aac_state_test_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_test.npy
CONFIG=$BASEDIR/environments/test.cfg
INSTANCE=oblige

if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 20 \
    --batch_size 20 \
    --episode_discount 0.95 \
    --model $MODEL \
    --state_model $STATE_MODEL \
    --action_set $ACTION_SET \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100 \
    --episode_num 50000
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main.py \
    --mode train \
    --episode_size 20 \
    --batch_size 20 \
    --episode_discount 0.95 \
    --model $MODEL \
    --state_model $STATE_MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100 \
    --episode_num 50000
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main.py \
    --mode test \
    --model $MODEL \
    --state_model $STATE_MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --doom_instance $INSTANCE \
    --vizdoom_config $CONFIG \
    --skiprate 1 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi