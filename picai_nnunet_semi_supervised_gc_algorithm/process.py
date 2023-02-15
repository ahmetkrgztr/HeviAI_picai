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

import glob
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from report_guided_annotation import extract_lesion_candidates
from tqdm import tqdm

from picai_baseline.nnunet.softmax_export import \
    save_softmax_nifti_from_softmax
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import PreprocessingSettings, Sample, crop_or_pad

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json


def get_json_as_dict(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def overwrite_affine(fixed_img: sitk.Image, moving_img: sitk.Image) -> sitk.Image:   
    moving_img.SetOrigin(fixed_img.GetOrigin())
    moving_img.SetDirection(fixed_img.GetDirection())
    moving_img.SetSpacing(fixed_img.GetSpacing())
    return moving_img

class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""

    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


def convert_to_original_extent(pred: np.ndarray, pkl_path: Union[Path, str], dst_path: Union[Path, str]):
    # convert to nnUNet's internal softmax format
    pred = np.array([1-pred, pred])

    # read physical properties of current case
    with open(pkl_path, "rb") as fp:
        properties = pickle.load(fp)

    # let nnUNet resample to original physical space
    save_softmax_nifti_from_softmax(
        segmentation_softmax=pred,
        out_fname=str(dst_path),
        properties_dict=properties,
    )


def extract_lesion_candidates_cropped(pred: np.ndarray, threshold: Union[str, float]):
    size = pred.shape
    pred = crop_or_pad(pred, (20, 384, 384))
    pred = crop_or_pad(pred, size)
    return extract_lesion_candidates(pred, threshold=threshold)[0]


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline nnU-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # input / output paths for algorithm
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri",
            "/input/images/transverse-adc-prostate-mri",
            "/input/images/transverse-hbv-prostate-mri",
        ]
        self.scan_paths = []
        self.cspca_detection_map_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_confidence_path = Path("/output/cspca-case-level-likelihood.json")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")
        
        self.noncroped_t2_empty_nifti_path = Path("/opt/algorithm/nnunet/noncroped_t2_empty_nifti_folder/t2_empty.nii.gz")
        
        self.nnunet_inp_dir_prostate = Path("/opt/algorithm/nnunet/input_prostate")
        self.nnunet_out_dir_prostate = Path("/opt/algorithm/nnunet/output_prostate")
    
        self.clinic_data_path = "/input/clinical-information-prostate-mri.json"
        
        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_inp_dir_prostate.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir_prostate.mkdir(exist_ok=True, parents=True)
        self.cspca_detection_map_path.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.image_input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # append scan path to algorithm input paths
                self.scan_paths += [file_paths[0]]

    def preprocess_input(self):
        """Preprocess input images to nnUNet Raw Data Archive format"""
        # set up Sample
        self.org_shape = sitk.ReadImage(self.scan_paths[0]).GetSize()[0]
        #if 750 < self.org_shape:
        if True:
            print("Cropped input")
            sample = Sample(
                scans=[
                    sitk.ReadImage(str(path))
                    for path in self.scan_paths
                ],
                settings=PreprocessingSettings(
                    physical_size=[81.0, 192.0, 192.0],
                    crop_only=True
                )
            )
        else:
            sample = Sample(
                scans=[
                    sitk.ReadImage(str(path))
                    for path in self.scan_paths
                ],
            )

        

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)
        
        src = self.nnunet_inp_dir / f"scan_0000.nii.gz"
        dst = self.nnunet_inp_dir_prostate / f"scan_0000.nii.gz"
        shutil.copyfile(src, dst)
        

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and generate detection map for clinically significant prostate cancer
        """
        
        start_time = time.time()
        # perform preprocessing
        self.preprocess_input()
        
        self.predict(
                task="Task848_experiment48",
                trainer="nnUNetTrainerV2_MMS",
                checkpoint="model_best",
                nnunet_inp_dir = str(self.nnunet_inp_dir_prostate),
                nnunet_out_dir = str(self.nnunet_out_dir_prostate),
                folds="0"
            )
        
        pred_path_prostate = str(self.nnunet_out_dir_prostate / "scan.npz")
        sm_arr = np.load(pred_path_prostate)['softmax']
        pz_arr = np.array(sm_arr[1,:,:,:]).astype('float32')
        sz_arr = np.array(sm_arr[2,:,:,:]).astype('float32')
        
        convert_to_original_extent(
            pred=pz_arr,
            pkl_path=self.nnunet_out_dir_prostate / "scan.pkl",
            dst_path=self.nnunet_out_dir_prostate / "softmax_pz.nii.gz",
        )
        
        convert_to_original_extent(
            pred=sz_arr,
            pkl_path=self.nnunet_out_dir_prostate / "scan.pkl",
            dst_path=self.nnunet_out_dir_prostate / "softmax_sz.nii.gz",
        )
        
        t2_sitk = sitk.ReadImage(self.nnunet_inp_dir / f"scan_0000.nii.gz")
        pz_sitk = sitk.ReadImage(self.nnunet_out_dir_prostate / "softmax_pz.nii.gz")
        sz_sitk = sitk.ReadImage(self.nnunet_out_dir_prostate / "softmax_sz.nii.gz")
        
        pz_sitk = overwrite_affine(t2_sitk, pz_sitk)
        sz_sitk = overwrite_affine(t2_sitk, sz_sitk)
        
        pz_sitk.CopyInformation(t2_sitk)
        sz_sitk.CopyInformation(t2_sitk)
        
        #Calculate Prostate Volume and read PSA
        
        pz_arr = sitk.GetArrayFromImage(pz_sitk)
        sz_arr = sitk.GetArrayFromImage(sz_sitk)
        
        
        PZ_arr = np.array(pz_arr > 0.5).astype("int")
        SZ_arr = np.array(sz_arr > 0.5).astype("int")
        
        ttl_pixel = np.sum(PZ_arr) + np.sum(SZ_arr)
        x,y,z = pz_sitk.GetSpacing()
        
        prostate_volume = (ttl_pixel*x*y*z)/(1000)
        
        try:
            psa = get_json_as_dict(self.clinic_data_path)["PSA_report"]
            psa_ratio = psa/prostate_volume
        except Exception as e:
            print("\n", e, "\n")
            psa_ratio=10
            
        print("\n psa_ratio", psa_ratio, "prostate_volume", prostate_volume, "\n")
            
            
        pz_save_path = self.nnunet_inp_dir / f"scan_0003.nii.gz"
        sz_save_path = self.nnunet_inp_dir / f"scan_0004.nii.gz"
        
        atomic_image_write(pz_sitk, pz_save_path)
        atomic_image_write(sz_sitk, sz_save_path)
        
        if psa_ratio >= 0.04:
            # perform inference using nnUNet
            pred_ensemble = None
            ensemble_count = 0
            for trainer in [
                "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
            ]:
                # predict sample
                self.predict(
                    task="Task2203_picai_baseline",
                    trainer=trainer,
                    checkpoint="model_best",
                    nnunet_inp_dir = str(self.nnunet_inp_dir),
                    nnunet_out_dir = str(self.nnunet_out_dir),
                    folds="0,1,2,3,4"
                )

                # read softmax prediction
                pred_path = str(self.nnunet_out_dir / "scan.npz")
                pred = np.array(np.load(pred_path)['softmax'][1]).astype('float32')
                os.remove(pred_path)
                if pred_ensemble is None:
                    pred_ensemble = pred
                else:
                    pred_ensemble += pred
                ensemble_count += 1

            # average the accumulated confidence scores
            pred_ensemble /= ensemble_count

            # the prediction is currently at the size and location of the nnU-Net preprocessed
            # scan, so we need to convert it to the original extent before we continue
            convert_to_original_extent(
                pred=pred_ensemble,
                pkl_path=self.nnunet_out_dir / "scan.pkl",
                dst_path=self.nnunet_out_dir / "softmax.nii.gz",
            )

            # now each voxel in softmax.nii.gz corresponds to the same voxel in the original (T2-weighted) scan
            pred_ensemble = sitk.ReadImage(str(self.nnunet_out_dir / "softmax.nii.gz"))

            # extract lesion candidates from softmax prediction
            # note: we set predictions outside the central 81 x 192 x 192 mm to zero, as this is far outside the prostate
            detection_map = extract_lesion_candidates_cropped(
                pred=sitk.GetArrayFromImage(pred_ensemble),
                threshold="dynamic"
            )

            # convert detection map to a SimpleITK image and infuse the physical metadata of original T2-weighted scan
            self.org_shape = sitk.ReadImage(self.scan_paths[0]).GetSize()[0]
            
            reference_scan_original_path = str(self.scan_paths[0])
            reference_scan_original = sitk.ReadImage(reference_scan_original_path)
            #detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
            #detection_map.CopyInformation(reference_scan_original)
            ref_shape = reference_scan_original.GetSize()
            detection_map = crop_or_pad(detection_map, (ref_shape[-1], ref_shape[1], ref_shape[0]))

            adc_path = str(self.nnunet_inp_dir / "scan_0001.nii.gz")
            adc_sitk = sitk.ReadImage(adc_path)
            adc_arr = sitk.GetArrayFromImage(adc_sitk)
            adc_arr = crop_or_pad(adc_arr, (ref_shape[-1], ref_shape[1], ref_shape[0]))
            adc_arr[detection_map == 0] = 0
            case_list = list(np.unique(adc_arr))
            case_list.remove(0)
            case_list.sort()
            list_len = len(case_list)
            adc_val = np.mean(case_list[0:list_len//10])
            print(adc_val)

            detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
            detection_map.CopyInformation(reference_scan_original)


            if adc_val > 1200:
            #if True:
                with open(self.case_confidence_path, 'w') as fp:
                    json.dump(float(0.0), fp)
            
                #reference_scan_original_path = str(self.scan_paths[0])
                #reference_scan_original = sitk.ReadImage(reference_scan_original_path)

                detection_map: sitk.Image = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(reference_scan_original).shape))
                detection_map.CopyInformation(reference_scan_original)
                atomic_image_write(detection_map, str(self.cspca_detection_map_path))
                
            else:
                # save prediction to output folder
                atomic_image_write(detection_map, str(self.cspca_detection_map_path))

                # save case-level likelihood
                with open(self.case_confidence_path, 'w') as fp:
                    val = np.max(sitk.GetArrayFromImage(detection_map))
                    json.dump(float(val), fp)
        else:
            with open(self.case_confidence_path, 'w') as fp:
                json.dump(float(0.0), fp)
            
            reference_scan_original_path = str(self.scan_paths[0])
            reference_scan_original = sitk.ReadImage(reference_scan_original_path)
            
            detection_map: sitk.Image = sitk.GetImageFromArray(np.zeros(sitk.GetArrayFromImage(reference_scan_original).shape))
            detection_map.CopyInformation(reference_scan_original)
            atomic_image_write(detection_map, str(self.cspca_detection_map_path))

        print("T2 shape", sitk.GetArrayFromImage(reference_scan_original).shape)
        print("Detectionmap shape", sitk.GetArrayFromImage(sitk.ReadImage(str(self.cspca_detection_map_path))).shape)
        print("--- Execution time %s seconds ---" % (time.time() - start_time))
            
    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False, nnunet_inp_dir = None, nnunet_out_dir = None):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)
        os.environ['MKL_SERVICE_FORCE_INTEL'] = "GNU"
        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', nnunet_inp_dir,
            '-o', nnunet_out_dir,
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '4',
            '--num_threads_nifti_save', '2'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    csPCaAlgorithm().process()
