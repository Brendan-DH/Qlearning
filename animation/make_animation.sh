#!/bin/bash

FILENAME=$1
OUTPUTNAME=$2
FRAMERATE=$3

echo $FILENAME
echo $OUTPUTNAME

ffmpeg -i $FILENAME -vf palettegen palette.png
ffmpeg -framerate $FRAMERATE -i $FILENAME -i palette.png -filter_complex "[0:v][1:v] paletteuse" -y  $OUTPUTNAME
