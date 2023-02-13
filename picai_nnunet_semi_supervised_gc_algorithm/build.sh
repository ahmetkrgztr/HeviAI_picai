#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t picai_baseline_nnunet_semi_supervised_processor:v2.1.1 \
    -t picai_baseline_nnunet_semi_supervised_processor:latest
