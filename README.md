## 
<h1 align="center"><span style="color:#5088B8">The Latent Color Subspace:</span> Emergent Order in High-Dimensional Chaos</h1>

<div align="center">
<a href="https://mateuszpach.github.io/">Mateusz Pach*</a>,
<a href="https://jessica-bader.github.io/">Jessica Bader*</a>,
<a href="https://qbouniot.github.io/">Quentin Bouniot</a>,
<a href="https://sergebelongie.github.io/">Serge Belongie</a>,
<a href="https://www.eml-munich.de/people/zeynep-akata">Zeynep Akata</a>
<br>
<span style="font-size:13px"> * denotes equal contribution </span>
<br>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2603.12261)
</div>

<h3 align="center">Abstract</h3>

<p align="justify">
     Text-to-image generation models have advanced rapidly, yet achieving fine-grained control over generated images remains difficult, largely due to limited understanding of how semantic information is encoded.
     We develop an interpretation of the color representation in the Variational Autoencoder latent space of FLUX.1 [Dev], revealing a structure reflecting Hue, Saturation, and Lightness.
     We verify our Latent Color Subspace (LCS) interpretation by demonstrating that it can both predict and explicitly control color, introducing a fully training-free method in FLUX based solely on closed-form latent-space manipulation.
</p>
<br>
<div align="center">
    <img src="assets/teaser.svg" alt="Method" width="400">
</div>


## Code
This repository contains code for identifying, observing, and intervening on the LCS, along with evaluation scripts for reproducing the experiments presented in the paper.

#### Environment
```
conda create -n lcs python=3.10 pip=25.1
conda activate lcs
pip install -r requirements.txt
```

### Finding the LCS
To find the LCS run the following:
```
cd find_lcs/
python save_hsv_latents.py
python save_geometry_with_anchors.py
python save_alphas_and_betas.py
```
To plot the trajectories and the LCS run:
```
python plot_walls.py
python plot_lcs.py
```

### Observing the LCS
Observation experiments can be reproduced as follows:
```
cd observation/
python generate_quali.py
python generate_objects.py
python generate_walls.py
python evaluate.py --mode objects
python evaluate.py --mode walls
```

### Intervening with the LCS

#### General Intervention
A general LCS intervention can be applied to any target prompt using `generate_images.py`. First, the hex value and color name must be added to the dictionary COLORS (some color examples can already be found there). The file takes the following most important arguments: 

- `out_path` (`str`): Path to the final image, including extention (e.g., .png).
- `prompt` (`str`): The target prompt.
- `objects` (`str`): list of target objects to be modified, separate by commas. Paired with colors, should be same length and in same order.
- `colors` (`str`): list of target colors to modify each object to, separate by commas. Paired with objects, should be same length and in same order.
- `replace_timestep` (`int`): The timestep to make the intervention.
- `save_visualizations` (`store_true`): Set to true if intermediate steps should be saved.
- `save_dir` (`str`): The output directory for visualizations.
- `remove_color` (`store_true`): Set this to true if all color words should be removed from the prompt.
- `global_mod` (`store_true`): Set this to true if the modifcation should be global. Otherwise will be local.

For example, to run a single prompt with local interventions:
```
python generate_images.py \
        --out_path /path/to/output/teddy_bear.png \
        --prompt "a photo of a teddy bear" \
        --objects "teddy bear" \
        --colors "red" \
        --replace_timestep 9 \
        --seed 0
```

#### Generate for Paper Results
To generate images for the results in the paper, scripts are included with the specific lists of prompts and colors necessary for each set. These also allow ints `--split_start` and `--split_end` to be designated, so the dataset generation may be parallelized.

Precise Natural global:
```
python generate_precise_natural.py \
    --out_path "/path/to/folder/imgs/folder_name1" \
    --replace_timestep 9 \
    --split_start "0" \
    --split_end "52" \
    --global_mod
```
Precise Plain:
```
python generate_precise_natural.py \
    --out_path "/path/to/folder/imgs/folder_name1" \
    --replace_timestep 9 \
    --split_start "0" \
    --split_end "52" \
```
Precise Natural small local:
```
python generate_precise_natural_small.py \
    --out_path "/path/to/folder/imgs/folder_name1" \
    --replace_timestep 9 \
    --split_start "0" \
    --split_end "52" \
```
GenEval:
```
python generate_geneval.py \
    --out_path "/path/to/folder/imgs/folder_name1" \
    --replace_timestep 9 \
    --remove_color
```
#### Note on the Core Files
The intervention itself takes place in `flux_replace_latent.py`. `flux_visualize_attention.py` is necessary for saving the attention maps used for local intervention.


### Evaluation Scripts
Evaluation scripts can be found in the `evaluation/` folder. Because the precise benchmarks leverage GenEval's object detector, everything must be run in the GenEval environment (we invite the reader to view GenEval for instructions).

#### ΔColor Step 1: Extract Masks
To measure ΔE00, ΔH, ΔS, and ΔL for the precise datasets with the provided scripts, all target images to evaluate at once must be in a single folder (e.g., `/path/to/folder/imgs/folder_name1`, see below). Then, masks must be generated for the target data with `evaluation/extract_masks.py`, which will indicate where the target object is (this step may be skipped for the plain benchmark). The file must be modified to indicate where `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py` (`MODEL_CONFIG`) and `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth` (`MODEL_CHECKPOINT`) are locally. `CLASSNAMES_PATH` should point to `object_names.txt`, also found in the evaluation folder. Set `ROOT` to `/path/to/folder/imgs/folder_name1`. Finally, set `DET_MASK_OUT_ROOT` to the desired output folder for the mask, which must have the same final directory (e.g., `folder_name1`) as `ROOT` (e.g., `/path/to/folder/masks/folder_name1`, see below). Run `extract_masks.py`.

```
Example structure:
├── imgs
│   ├── folder_name1
│   ├── folder_name2
│   └── ...
├── masks
│   ├── folder_name1
│   ├── folder_name2
│   └── ...
```

#### ΔColor Step 2: Measure
Then, ΔE00, ΔH, ΔS, and ΔL can be measured with `evaluation/evaluate_precise.py`. Put `img_folder` in `/path/to/folder/imgs`, `mask_folder` in `/path/to/folder/masks/`, and set `v` to `folder_name1`. If measuring Precise Plain, use `no_make = False`, otherwise `no_mask = True`. Additionally, designate a folder to print the results to in `save_path`. The, run `evaluation/evaluate_precise.py`.

# Citation
```bibtex
@article{pach2026latentcolorsubspace,
  title={The Latent Color Subspace: Emergent Order in High-Dimensional Chaos}, 
  author={Mateusz Pach and Jessica Bader and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={International Conference on Machine Learning},
  year={2026}
}
```