#!/bin/bash
datasets=( mls_spanish_opus mls_dutch_opus mls_french_opus mls_italian_opus mls_german_opus mls_polish_opus mls_portuguese_opus )

for dataset in "${datasets[@]}"; do
  LIBRISPEECH_PATH="/path/to/$dataset"
  DATASET_OUT_PATH="/path/to/outdir"

  OUT_PATH=$DATASET_OUT_PATH/${dataset}8k
  echo $OUT_PATH
  if [[ ! -e $OUT_PATH ]]; then
      mkdir $OUT_PATH
  fi
  splits=( train test dev )
  for s in "${splits[@]}"; do
    echo "Split $s"
    if [[ ! -e $OUT_PATH/$s ]]; then
      mkdir $OUT_PATH/$s
    fi
    for o in $LIBRISPEECH_PATH/$s/audio/*/; do
    #echo "Dir: $o"
    for d in $o/*/; do
     #echo "Dir: $d"
     for f in $d/*.opus; do
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
done
