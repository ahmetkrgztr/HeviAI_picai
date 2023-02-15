docker run --cpus=16 --memory=32g --shm-size=32g --rm -it \
    -v /media/hdd2/works/HeviAI_picai/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code:ro \
    -v /media/hdd2/works/HeviAI_picai/debug_workdir:/workdir \
    -v /media/hdd2/works/HeviAI_picai/input_images:/input/images:ro \
    -v /media/hdd2/works/HeviAI_picai/picai_labels:/input/picai_labels:ro \
    -v /media/hdd2/works/HeviAI_picai/debug_checkpoints:/checkpoints \
    -v /media/hdd2/works/HeviAI_picai/debug_preprocessed:/output \
    joeranbosma/picai_nnunet:latest \
    python /code/preprocess.py --splits picai_pub
