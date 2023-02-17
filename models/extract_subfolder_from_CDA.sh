#!/bin/bash
name="C-CDA"
for folder in "C-CDA"/*; do
#  echo foldername=""
  cp -r "$folder" "C-CDA-${folder##*/}"
done