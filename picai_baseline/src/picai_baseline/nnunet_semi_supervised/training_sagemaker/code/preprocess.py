#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import json
import os
import shutil
from pathlib import Path
from subprocess import check_call
import os

from picai_prep.preprocessing import Sample, crop_or_pad, PreprocessingSettings
import SimpleITK as sitk
from picai_prep.data_utils import atomic_image_write
from tqdm import tqdm

from picai_baseline.prepare_data_semi_supervised import \
    prepare_data_semi_supervised
from picai_baseline.splits.picai import nnunet_splits as picai_pub_splits
from picai_baseline.splits.picai_debug import \
    nnunet_splits as picai_debug_splits
from picai_baseline.splits.picai_nnunet import \
    nnunet_splits as picai_pub_nnunet_splits
from picai_baseline.splits.picai_pubpriv import \
    nnunet_splits as picai_pubpriv_splits
from picai_baseline.splits.picai_pubpriv_nnunet import \
    nnunet_splits as picai_pubpriv_nnunet_splits

class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super.__init__(message)
        
def overwrite_affine(fixed_img: sitk.Image, moving_img: sitk.Image) -> sitk.Image:   
    moving_img.SetOrigin(fixed_img.GetOrigin())
    moving_img.SetDirection(fixed_img.GetDirection())
    moving_img.SetSpacing(fixed_img.GetSpacing())
    #moving_img.CopyInformation(fixed_img)
    return moving_img


def main(taskname="Task2203_picai_baseline"):
    """Train nnU-Net semi-supervised model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--splits', type=str, default="picai_pubpriv",
                        help="Cross-validation splits. Can be a path to a json file or one of the predefined splits: "
                             "picai_pub, picai_pubpriv, picai_pub_nnunet, picai_pubpriv_nnunet, picai_debug.")
    parser.add_argument('--nnUNet_tf', type=int, default=8, help="Number of preprocessing threads for full images")
    parser.add_argument('--nnUNet_tl', type=int, default=8, help="Number of preprocessing threads for low-res images")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    splits_path = workdir / f"splits/{taskname}/splits.json"

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    # set environment variables
    os.environ["prepdir"] = str(workdir / "nnUNet_preprocessed")

    # set nnU-Net's number of preprocessing threads
    os.environ["nnUNet_tf"] = str(args.nnUNet_tf)
    os.environ["nnUNet_tl"] = str(args.nnUNet_tl)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")

    print("Images folder:", os.listdir(images_dir))
    print("Labels folder:", os.listdir(labels_dir))

    # resolve cross-validation splits
    predefined_splits = {
        "picai_pub": picai_pub_splits,
        "picai_pubpriv": picai_pubpriv_splits,
        "picai_pub_nnunet": picai_pub_nnunet_splits,
        "picai_pubpriv_nnunet": picai_pubpriv_nnunet_splits,
        "picai_debug": picai_debug_splits,
    }
    if args.splits in predefined_splits:
        nnunet_splits = predefined_splits[args.splits]
    else:
        # `splits` should be the path to a json file containing the splits
        print(f"Loading splits from {args.splits}")
        with open(args.splits, "r") as f:
            nnunet_splits = json.load(f)

    # save cross-validation splits to disk
    with open(splits_path, "w") as fp:
        json.dump(nnunet_splits, fp)

    # Convert MHA Archive to nnU-Net Raw Data Archive
    # Also, we combine the provided human-expert annotations with the AI-derived annotations.
    print("Preprocessing data...")
    prepare_data_semi_supervised(
        workdir=workdir,
        imagesdir=images_dir,
        labelsdir=labels_dir,
        #preprocessing_kwargs='{"physical_size": [81.0, 192.0, 192.0], "crop_only": true}',
        splits="picai_pubpriv",
    )
    
    #Check if is there any missing in softmax predictions
    #If not add softmax predictions as additional modalities
    
    all_case_names = nnunet_splits[0]["train"] + nnunet_splits[0]["val"]
    all_case_names.sort()
    
    for case_name in all_case_names:
        for zone in ["pz", "tz"]:
            case_zone_sm_path = os.path.join(labels_dir, f"anatomical_delineations/zonal_pz_tz/AI/HeviAI23/{case_name}/prostate_gland_sm_{zone}.nii.gz")
            if not os.path.exists(case_zone_sm_path):
                raise MissingSequenceError(name=case_zone_sm_path.split("/")[-1], folder=case_zone_sm_path)

    
    print("Preparing prostate gland softmax predictions..")
    for case_name in tqdm(all_case_names):
        case_sm_paths = []
        for zone in ["pz", "tz"]:
            case_zone_sm_path = os.path.join(labels_dir, f"anatomical_delineations/zonal_pz_tz/AI/HeviAI23/{case_name}/prostate_gland_sm_{zone}.nii.gz")
            case_sm_paths.append(case_zone_sm_path)

        sample = Sample(
                    scans = [sitk.ReadImage(str(path)) for path in case_sm_paths],
                    #settings = PreprocessingSettings(
                        #physical_size = [81.0, 192.0, 192.0],
                        #crop_only = True
                    #)
                )
        sample.preprocess()

        t2_sitk_img = sitk.ReadImage(os.path.join(workdir, "nnUNet_raw_data/Task2203_picai_baseline/imagesTr", f"{case_name}_0000.nii.gz"))
        #ref_shape = t2_sitk_img.GetSize()
        for i, scan in enumerate(sample.scans):
            path = os.path.join(workdir, "nnUNet_raw_data/Task2203_picai_baseline/imagesTr", f"{case_name}_{i+3:04d}.nii.gz")
            #scan_arr = sitk.GetArrayFromImage(scan)
            #scan_arr = crop_or_pad(scan_arr, (ref_shape[-1], ref_shape[1], ref_shape[0]))
            #scan = sitk.GetImageFromArray(scan_arr)
            overwritten_scan = overwrite_affine(t2_sitk_img, scan)
            atomic_image_write(overwritten_scan, path)
    
    #Update dataset json
    dataset_json_path = os.path.join(workdir, f"nnUNet_raw_data/{taskname}/dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
        
    os.remove(dataset_json_path)
    
    dataset_json["modality"] = {'0': 'T2W', '1': 'CT', '2': 'HBV', '3': 'PZ', '4': 'TZ'}
    with open(dataset_json_path, "w") as fp:
        json.dump(dataset_json, fp)
       
    # Preprocess data with nnU-Net
    print("Preprocessing data with nnU-Net...")
    cmd = [
        "nnunet", "plan_train", str(taskname), workdir.as_posix(),
        "--custom_split", str(splits_path),
        "--plan_only",
    ]
    check_call(cmd)

    # Export preprocessed dataset
    print("Exporting preprocessed dataset...")
    dst = output_dir / f"nnUNet_preprocessed/{taskname}/"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workdir / f"nnUNet_preprocessed/{taskname}/", dst)


if __name__ == '__main__':
    main()
