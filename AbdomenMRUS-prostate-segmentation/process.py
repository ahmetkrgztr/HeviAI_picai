import os
import pickle
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep import atomic_file_copy, atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)

from picai_baseline.nnunet.softmax_export import \
    save_softmax_nifti_from_softmax


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


def strip_metadata(img: sitk.Image) -> None:
    for key in img.GetMetaDataKeys():
        img.EraseMetaData(key)


def overwrite_affine(fixed_img: sitk.Image, moving_img: sitk.Image) -> sitk.Image:   
    moving_img.SetOrigin(fixed_img.GetOrigin())
    moving_img.SetDirection(fixed_img.GetDirection())
    moving_img.SetSpacing(fixed_img.GetSpacing())
    return moving_img


class ProstateSegmentationAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained prostate segmentation nnU-Net model from
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
        self.input_dirs = [
            "/input/images/transverse-t2-prostate-mri"
        ]
        self.scan_paths = []
        self.prostate_segmentation_path_pz = Path("/output/images/softmax-prostate-peripheral-zone-segmentation/prostate_gland_sm_pz.mha")
        self.prostate_segmentation_path_tz = Path("/output/images/softmax-prostate-central-gland-segmentation/prostate_gland_sm_tz.mha")
        self.prostate_segmentation_path = Path("/output/images/prostate-zonal-segmentation/prostate_gland.mha")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")

        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.prostate_segmentation_path_pz.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.input_dirs:
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
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in [self.scan_paths[0]]
            ],
            settings=PreprocessingSettings(
                physical_size=[81.0, 192.0, 192.0],
                crop_only=True
            )
        )

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and segment the prostate glands
        """
        # perform preprocessing
        self.preprocess_input()

        # perform inference using nnUNet
        self.predict(
                task="Task848_experiment48",
                trainer="nnUNetTrainerV2_MMS",
                checkpoint="model_best",
                folds="0"
            )

        pred_path_prostate = str(self.nnunet_out_dir / "scan.npz")
        sm_arr = np.load(pred_path_prostate)['softmax']
        pz_arr = np.array(sm_arr[1, :, :, :]).astype('float32')
        tz_arr = np.array(sm_arr[2, :, :, :]).astype('float32')

        # read postprocessed prediction
        pred_path = str(self.nnunet_out_dir / "scan.nii.gz")
        pred: sitk.Image = sitk.ReadImage(pred_path)

        # save postprocessed prediction to output
        atomic_file_copy(pred_path, str(self.prostate_segmentation_path))

        for pred, save_path in [
            (pz_arr, self.prostate_segmentation_path_pz),
            (tz_arr, self.prostate_segmentation_path_tz),
        ]:
            # the prediction is currently at the size and location of the nnU-Net preprocessed
            # scan, so we need to convert it to the original extent before we continue
            convert_to_original_extent(
                pred=pred,
                pkl_path=self.nnunet_out_dir / "scan.pkl",
                dst_path=self.nnunet_out_dir / "softmax.nii.gz",
            )

            # now each voxel in softmax.nii.gz corresponds to the same voxel in the reference scan
            pred = sitk.ReadImage(str(self.nnunet_out_dir / "softmax.nii.gz"))

            # convert prediction to a SimpleITK image and infuse the physical metadata of the reference scan
            reference_scan_original_path = str(self.scan_paths[0])
            reference_scan = sitk.ReadImage(reference_scan_original_path)
            pred = resample_to_reference_scan(pred, reference_scan_original=reference_scan)
            
            # clip small values to 0 to save disk space
            arr = sitk.GetArrayFromImage(pred)
            arr[arr < 1e-3] = 0
            pred_clipped = sitk.GetImageFromArray(arr)
            pred_clipped.CopyInformation(pred)

            # remove metadata to get rid of SimpleITK warning
            strip_metadata(pred_clipped)

            # save prediction to output folder
            atomic_image_write(pred_clipped, save_path, mkdir=True)

    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_inp_dir),
            '-o', str(self.nnunet_out_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
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
    ProstateSegmentationAlgorithm().process()
