# EUSR-GAN: ultrasound super-resolution and the effect of interpolated references

[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24568376.svg)](https://doi.org/10.5281/zenodo.24568376)

Code for the manuscript *"EUSR-GAN: A lightweight multi-scale GAN for ultrasound super-resolution and the effect of interpolated references on its evaluation"* (submitted).

EUSR-GAN is a 2.33 M-parameter generative adversarial network for ×4 super-resolution (SR) of B-mode ultrasound. It builds on the multi-scale, progressive-upscaling design of Ahmad et al. (Sci. Rep. 2022), adding:

- dilated and depthwise-separable branches;
- squeeze-and-excitation attention;
- a residual-in-residual dense trunk;
- group normalization;
- a Fourier-domain loss.

The repository contains the code for all the experiments in the manuscript:

- the comparison with the protocol of Ahmad et al. on CAMUS;
- the native-resolution, region-wise evaluation on CAMUS and BUSI;
- the learned baselines (EDSR-baseline, SwinIR-light);
- ablations, cross-organ transfer and task-based evaluation (lesion detection and segmentation);
- the wavelet sub-band analysis.

It also includes a web prototype that runs the trained model.

> Research code. The models are not medical devices and must not be used for diagnosis.

## Repository layout

```
experiments/
  camus_eusr.py          EUSR-GAN on CAMUS: Ahmad protocol (HR_MODE='ahmad') or native (HR_MODE='native')
  ahmad_camus.py         re-implementation of Ahmad et al. (2022) on the same CAMUS split
  busi_native.py         EUSR-GAN on BUSI at native resolution (main BUSI experiment; seeds, CAMUS transfer)
  baselines_sr.py        EDSR-baseline and SwinIR-light in Keras (same input/output as EUSR-GAN)
  busi_baseline.py       trains a baseline on BUSI with exactly the EUSR-GAN pipeline (ARCH, SEED, RESUME)
  camus_baseline.py      same for CAMUS (ARCH=eusr|edsr|swinir, SEED, RESUME)
  busi_tasks.py          task-based evaluation on BUSI: YOLOv8n detection, U-Net segmentation, hallucinations
  busi_wavelet.py        wavelet sub-band energy and error by region (inference only)
  figuras_articulo.py    generates the manuscript figures from the saved weights and per-image results
  ablations/
    busi_nofft.py          without the Fourier loss
    busi_residual.py       residual output over bicubic upsampling
    busi_rrdb8.py          8 RRDBs instead of 4
    busi_self_ensemble.py  ×8 self-ensemble at test time (EUSR-GAN+), no retraining
    ahmad_camus_no_gan.py  Ahmad et al. without the adversarial stage
  legacy_v6/             earlier BUSI pipeline at 256×256 (reported in the Discussion only)
web/                     web prototype (FastAPI + browser UI): upload an image, mark a region, compare
tools/
  zenodo_subir_resultados.py  packages trained weights and per-image results and uploads them to Zenodo
```

The ablation scripts are standalone copies of `busi_native.py` with one setting changed (see the header of each file), so every reported run can be reproduced as is.

## Data

Both datasets are public and are **not** redistributed here:

| Dataset | Source | Expected path |
|---|---|---|
| CAMUS (echocardiography, 500 patients) | Leclerc et al., IEEE TMI 2019 — https://www.creatis.insa-lyon.fr/Challenge/camus/ | `camus/database_nifti/` |
| BUSI (breast ultrasound, 780 images) | Al-Dhabyani et al., Data in Brief 2020 | `busi/Dataset_BUSI_with_GT/` |

Splits are fixed and created by the scripts themselves:

- **CAMUS:** split by patient, 360/40/100, seed 42.
- **BUSI:** after removing near-duplicates, stratified by class, 616/78/78, seed 42.

## Environment

The experiments were run on a CPU-only cloud machine (8 cores, 128 GB RAM) with Python 3.11 and TensorFlow 2.12.

```bash
pip install -r requirements.txt
```

TensorFlow 2.12 requires NumPy < 1.24. Installing a newer package that upgrades NumPy (e.g. the latest PyWavelets) breaks TensorFlow, so keep the pinned versions.

## Reproducing the experiments

Run every script from the `experiments/` folder (they import each other and write results next to them).

```bash
cd experiments

# CAMUS — protocol of Ahmad et al. (HR resized to 1024×512) and native protocol
python -u camus_eusr.py                     # HR_MODE='ahmad' (edit to 'native' for the native run)
python -u ahmad_camus.py                    # re-implementation of Ahmad et al. on the same split

# BUSI — native resolution (seeds 42, 7, 123: edit TRAIN_SEED)
python -u busi_native.py

# Learned baselines with the same pipeline
ARCH=edsr   SEED=42 python -u busi_baseline.py
ARCH=swinir SEED=42 python -u busi_baseline.py
ARCH=swinir RESUME=1 python -u busi_baseline.py        # continue until validation SSIM plateaus
ARCH=edsr   SEED=7  python -u camus_baseline.py        # ARCH=eusr|edsr|swinir, SEED=42|7|123

# Ablations, task-based evaluation, wavelet analysis and figures
python -u ablations/busi_nofft.py           # (and the other files in ablations/)
python -u busi_tasks.py
python -u busi_wavelet.py
python -u figuras_articulo.py
```

Each run writes a `resultados_*` folder with:

- the best weights;
- training logs;
- a JSON file with the configuration, summary metrics and **per-image** results;
- paired Wilcoxon tests against the reference model when one is given.

Typical training time on the CPU machine is 3.5–4.5 h per EUSR-GAN run.

## Trained weights and per-image results

The trained weights (`*.weights.h5`), the per-image result files (`resultados_*.json`), training logs and the YOLOv8 lesion detector that underlie every table and figure are available on Zenodo: **[doi:10.5281/zenodo.24568376](https://doi.org/10.5281/zenodo.24568376)**.

## Web prototype

`web/` contains a FastAPI application that loads the trained weights and lets a user:

- upload an image (PNG, JPG, TIFF or DICOM);
- mark a region;
- compare EUSR-GAN with bicubic interpolation (slider, side by side, error maps).

It offers two modes:

- **×4 enlargement of the region (exploratory):** the region is enlarged as acquired; there is no reference to measure the improvement.
- **Validation:** the region is downsampled ×4 and reconstructed, and PSNR/SSIM are measured against the original, as in the manuscript.

See `web/README.md` (in Spanish) for local use and deployment with Docker.

## Citation

If you use this code, please cite the manuscript (reference to be updated on publication):

```
Osorio Barone A, Contreras S, Puertas E. EUSR-GAN: A lightweight multi-scale GAN for
ultrasound super-resolution and the effect of interpolated references on its evaluation.
[Journal]. [Year].
```

## Authors

- Alexander Osorio Barone ([ORCID 0009-0003-7537-8294](https://orcid.org/0009-0003-7537-8294)), Doctorate in Engineering, Universidad Tecnológica de Bolívar, Cartagena, Colombia
- Sonia Contreras (advisor), Universidad Tecnológica de Bolívar
- Edwin Puertas (co-advisor), Universidad Tecnológica de Bolívar

## License

MIT (see `LICENSE`). The CAMUS and BUSI datasets are subject to their own terms of use.
