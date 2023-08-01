#!/bin/bash
MSD_PATH="/path/to/millionsong8k"
dir_size=10000
if [[ ! -e "$MSD_PATH/train" ]]; then
    mkdir "$MSD_PATH/train"
fi
if [[ ! -e "$MSD_PATH/test" ]]; then
    mkdir "$MSD_PATH/test"
fi
if [[ ! -e $MSD_PATH"/valid" ]]; then
    mkdir "$MSD_PATH/valid"
fi
find $MSD_PATH -maxdepth 1 -type f | sort -R | head -1000 | xargs cp -t "$MSD_PATH/valid"
find $MSD_PATH -maxdepth 1 -type f | sort -R | head -10000 | xargs cp -t "$MSD_PATH/test"
find $MSD_PATH -maxdepth 1 -type f -print0 | xargs -0 mv -t "$MSD_PATH/train"