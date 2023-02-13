# PI-CAI: Baseline nnU-Net (semi-supervised)

## Managed By
Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands

## Contact Information
- Joeran Bosma: Joeran.Bosma@radboudumc.nl
- Anindo Saha: Anindya.Shaha@radboudumc.nl
- Henkjan Huisman: Henkjan.Huisman@radboudumc.nl

## Algorithm
This algorithm is hosted on [Grand-Challenge.com](https://grand-challenge.org/algorithms/pi-cai-baseline-nnu-net-semi-supervised/).

## Summary
This algorithm predicts a detection map for the likelihood of clinically significant prostate cancer (csPCa) using biparametric MRI (bpMRI). The algorithm ensembles 5-fold cross-validation models that were trained on the PI-CAI: Public Training and Development Dataset v2.0. The detection map is at the same spatial resolution and physical dimensions as the input axial T2-weighted image.

## Mechanism
This algorithm is a deep learning-based detection/diagnosis model, which ensembles 5 independent nnU-Net models  (5-fold cross-validation). To train these models, a total of 1500 prostate biparametric MRI (bpMRI) scans paired with human-annotated or AI-derived ISUP ≥ 2 delineations were used. See [picai_labels](https://github.com/DIAGNijmegen/picai_labels#annotations-and-resources) for details on the annotations. See [PI-CAI: Imaging Data](https://pi-cai.grand-challenge.org/DATA/) for details on the dataset.


**Source Code**: 

* [Algorithm inference](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm)
* [Algorithm training](https://github.com/DIAGNijmegen/picai_baseline/blob/main/nnunet_baseline.md#nnu-net---semi-supervised-training)

## Validation and Performance
_Pending._

## Uses and Directions
- **For research use only**. This algorithm is intended to be used only on biparametric prostate MRI examinations of patients with raised PSA levels or clinical suspicion of prostate cancer. This algorithm should not be used in different patient demographics. 

- **Benefits**: Risk stratification for clinically significant prostate cancer using prostate MRI is instrumental to reduce over-treatment and unnecessary biopsies. 

- **Target population**: This algorithm was trained on patients with raised PSA levels or clinical suspicion of prostate cancer, without prior treatment  (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), without prior positive biopsies, without artefacts and with reasonably-well aligned sequences. 

- **MRI scanner**: This algorithm was trained and evaluated exclusively on prostate bpMRI scans derived from Siemens Healthineers (Skyra/Prisma/Trio/Avanto) and Philips Medical Systems  (Ingenia/Achieva)  MRI scanners with surface coils. It does not account for vendor-neutral properties or domain adaptation, and in turn, is not compatible with scans derived using any other MRI scanner or those using endorectal coils.

- **Sequence alignment and position of the prostate**: While the input images (T2W, HBV, ADC) can be of different spatial resolutions, the algorithm assumes that they are co-registered or aligned reasonably well and that the prostate gland is localized within a volume of 460 cm³ from the centre coordinate.

- **General use**: This model is intended to be used by radiologists for predicting clinically significant prostate cancer in biparametric MRI examinations. The model is not a diagnostic for cancer and is not meant to guide or drive clinical care. This model is intended to complement other pieces of patient information in order to determine the appropriate follow-up recommendation.

- **Appropriate decision support**: The model identifies lesion X as at a high risk of being malignant. The referring radiologist reviews the prediction along with other clinical information and decides the appropriate follow-up recommendation for the patient.

- **Before using this model**: Test the model retrospectively and prospectively on a diagnostic cohort that reflects the target population that the model will be used upon to confirm the validity of the model within a local setting. 

- **Safety and efficacy evaluation**: To be determined in a clinical validation study.

## Warnings
- **Risks**: Even if used appropriately, clinicians using this model can misdiagnose cancer. Delays in cancer diagnosis can lead to metastasis and mortality. Patients who are incorrectly treated for cancer can be exposed to risks associated with unnecessary interventions and treatment costs related to follow-ups. 

- **Inappropriate Settings**: This model was not trained on MRI examinations of patients with prior treatment  (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), prior positive biopsies, artefacts or misalignment between sequences. Hence it is susceptible to faulty predictions and unintended behaviour when presented with such cases. Do not use the model in the clinic without further evaluation. 

- **Clinical rationale**: The model is not interpretable and does not provide a rationale for high risk scores. Clinical end users are expected to place the model output in context with other clinical information to make the final determination of diagnosis.

- **Inappropriate decision support**: This model may not be accurate outside of the target population. This model is not designed to guide clinical diagnosis and treatment for prostate cancer. 

- **Generalizability**: This model was primarily developed with prostate MRI examinations from Radboud University Medical Centre,  Andros Kliniek and University Medical Centre Groningen. Do not use this model in an external setting without further evaluation.

- **Discontinue use if**: Clinical staff raise concerns about the utility of the model for the intended use case or large, systematic changes occur at the data level that necessitates re-training of the model.
