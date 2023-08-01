#!/bin/bash
LIBRISPEECH_PATH="/path/to/librispeech/"
DATASET_OUT_PATH="/path/out"

OUT_PATH=$DATASET_OUT_PATH/LibriSpeech8k
if [[ ! -e $OUT_PATH ]]; then
    mkdir $OUT_PATH
fi
splits=( train-clean-360 test-clean dev-clean)
for s in "${splits[@]}"; do
  echo "Split $s"
  if [[ ! -e $OUT_PATH/$s ]]; then
    mkdir $OUT_PATH/$s
  fi
  for o in $LIBRISPEECH_PATH/$s/*/; do
  #echo "Dir: $o"
  for d in $o/*/; do
   #echo "Dir: $d"
   for f in $d/*.flac; do
    #echo $f
    [ -f "$f" ] || break
    filename=$(basename -- "$f")
    filename="${filename%.*}"
    #echo $filename
    ffmpeg -v error -y -i $f -ar 8000 -ac 1 -acodec pcm_s16le -af aresample=async=1 "$OUT_PATH/$s/$filename.wav"
   done
  done
  done
 done
