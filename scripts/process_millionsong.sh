#!/bin/bash
MSD_PATH="/path/to/millionsong-audio"
DATASET_OUT_PATH="/pat/out/"

OUT_PATH=$DATASET_OUT_PATH/millionsong8k
if [[ ! -e $OUT_PATH ]]; then
    mkdir $OUT_PATH
fi

for s in $MSD_PATH/mp3/*; do
  for o in $s/*; do
  #echo "Dir: $o"
  for d in $o/*; do
   #echo "Dir: $d"
   for f in $d/*.mp3; do

    [ -f "$f" ] || break
    filename=$(basename -- "$f")
    filename="${filename%.*}"
    #echo $filename
    #echo "$f"
    ffmpeg -v error -y -i "$f" -ar 8000 -ac 1 -acodec pcm_s16le -af aresample=async=1 "$OUT_PATH/$filename.wav"
   done
  done
  done
 done
