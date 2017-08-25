#!/usr/bin/env bash
BASEDIR=$(dirname "$( cd "$( dirname "$0" )" && pwd )")
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [train|resume|test]"
    exit 1
fi

COMMAND="$1"
MODEL=aac
CHECK_POINT=$BASEDIR/checkpoints/cig_server_aac_20_95_skip4_cp.pth
ACTION_SET=$BASEDIR/actions/action_set_speed_shot_backward_right.npy
CONFIG=$BASEDIR/environments/cig.cfg

BOT_CONFIG=$BASEDIR/environments/cig_server.cfg
BOT_MODEL=$BASEDIR/trained_models/cig_map01_aac_model.pth
BOT_CMD="python $BASEDIR/agent.py --vizdoom_config $BOT_CONFIG --action_set $ACTION_SET --model $BOT_MODEL"
#BOT_CMD="~/tools/vizdoom_cig2017/run.sh ~/tools/vizdoom_cig2017/doomnet_track1"


if [ $COMMAND == 'train' ]
then
    python $BASEDIR/src/main_train.py \
    --episode_size 20 \
    --batch_size 10 \
    --episode_discount 0.95 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $BASEDIR/trained_models/cig_map01_aac_model.pth \
    --vizdoom_config $CONFIG \
    --bot_cmd "$BOT_CMD" \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100
elif [ $COMMAND == 'resume' ]
then
    python $BASEDIR/src/main_train.py \
    --episode_size 20 \
    --batch_size 10 \
    --episode_discount 0.95 \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --bot_cmd "$BOT_CMD" \
    --skiprate 4 \
    --frame_num 1 \
    --checkpoint_file $CHECK_POINT \
    --checkpoint_rate 100
elif [ $COMMAND == 'test' ]
then
    python $BASEDIR/src/main_test.py \
    --model $MODEL \
    --action_set $ACTION_SET \
    --load $CHECK_POINT \
    --vizdoom_config $CONFIG \
    --skiprate 1 \
    --frame_num 1
else
    echo "'$COMMAND' is unknown command."
fi