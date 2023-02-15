#!/usr/bin/env bash

./build.sh

docker save heviai/picai_prostate_segmentation_processor:latest | gzip -c > picai_prostate_segmentation_processor-v1.0.tar.gz

