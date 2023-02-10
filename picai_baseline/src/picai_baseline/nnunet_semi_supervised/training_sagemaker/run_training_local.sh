docker run --cpus=16 --memory=61g --shm-size=61g --gpus='"device=1"' --rm -it \
    -v /media/hdd2/works/HeviAI_picai/picai_baseline/src/picai_baseline/nnunet_semi_supervised/training_sagemaker/code:/code:ro \
    -v /media/hdd2/works/HeviAI_picai/debug_workdir:/workdir \
    -v /media/hdd2/works/HeviAI_picai/input_images:/input/images:ro \
    -v /media/hdd2/works/HeviAI_picai/picai_labels:/input/picai_labels:ro \
    -v /media/hdd2/works/HeviAI_picai/debug_preprocessed:/input/preprocessed:ro \
    -v /media/hdd2/works/HeviAI_picai/debug-output:/output \
    -v /media/hdd2/works/HeviAI_picai/debug_checkpoints:/checkpoints \
    joeranbosma/picai_nnunet:latest \
    python /code/train.py
