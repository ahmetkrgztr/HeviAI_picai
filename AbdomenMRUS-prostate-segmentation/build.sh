#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t heviai/picai_prostate_segmentation_processor:latest \
    -t heviai/picai_prostate_segmentation_processor:v1.0
