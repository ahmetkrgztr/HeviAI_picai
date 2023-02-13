#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)

DOCKER_FILE_SHARE=picai_baseline_nnunet_semi_supervised_processor-output-$VOLUME_SUFFIX
docker volume create $DOCKER_FILE_SHARE
# you can see your output (to debug what's going on) by specifying a path instead:
# DOCKER_FILE_SHARE="/mnt/netcache/pelvis/projects/joeran/tmp-docker-volume"

docker run --cpus=4 --memory=32gb --shm-size=32gb --gpus='"device=0"' --rm \
        -v $SCRIPTPATH/test/:/input/ \
        -v $DOCKER_FILE_SHARE:/output/ \
        picai_baseline_nnunet_semi_supervised_processor

# check detection map (at /output/images/cspca-detection-map/cspca_detection_map.mha)
docker run --rm \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import numpy as np; import SimpleITK as sitk; f1 = sitk.GetArrayFromImage(sitk.ReadImage('/output/images/cspca-detection-map/cspca_detection_map.mha')); f2 = sitk.GetArrayFromImage(sitk.ReadImage('/input/cspca-detection-map/10032_1000032.mha')); print('N/o voxels more than 1e-3 differerent between prediction and reference:', np.sum(np.abs(f1-f2)>1e-3)); sys.exit(int(np.sum(np.abs(f1-f2)>1e-3)) > 10);"

if [ $? -eq 0 ]; then
    echo "Detection map test successfully passed..."
else
    echo "Expected detection map was not found..."
fi

# check case_confidence (at /output/cspca-case-level-likelihood.json)
docker run --rm \
        -v $DOCKER_FILE_SHARE:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        insighttoolkit/simpleitk-notebooks:latest python -c "import sys; import json; f1 = json.load(open('/output/cspca-case-level-likelihood.json')); f2 = json.load(open('/input/cspca-detection-map/10032_1000032.json')); print('Found case-level prediction %s, expected %s' % (f1, f2)); sys.exit(int(abs(f1-f2) > 1e-3));"


if [ $? -eq 0 ]; then
    echo "Case-level prediction test successfully passed..."
else
    echo "Expected case-level prediction was not found..."
fi

docker volume rm $DOCKER_FILE_SHARE

