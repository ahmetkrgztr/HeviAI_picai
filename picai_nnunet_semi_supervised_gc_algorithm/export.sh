#!/usr/bin/env bash

./build.sh

docker save picai_baseline_nnunet_semi_supervised_processor:latest | gzip -c > picai_baseline_nnunet_semi_supervised_processor_2.1.1.tar.gz
