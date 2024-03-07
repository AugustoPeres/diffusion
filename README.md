# Diffusion in jax

This repository contains the bare minimum necessary code to train a
simple diffusion model using
[jax](https://github.com/google/jax?tab=readme-ov-file) and
[equinox](https://github.com/patrick-kidger/equinox). We focus on
unconditional diffusion models and our implementation follows [Denoising Diffusion Probabilistic Models
](https://arxiv.org/abs/2006.11239) exactly.


## Running the code

The first thing you need to run this code is obviously the data. For
that just make sure that all your images are in the same
folder. Additionally they should already be cropped to the same
(square) size. Other than that there are no additional restrictions.

Next, create  virtual environment and install the requirements:

```bash
python3 -m venv .en
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
```


You are now ready to run the code:

```bash
python scripts/train.py --path_to_data=<path_to_your_images> \
                        --output_path=<where_to_save_artifact>
```

For the descriptions of the other flags run:

```bash
python scripts/train.py --help
```

## Samples on the MNIST dataset

The following images where generated from random noise after training
a model on the MNIST dataset.

|          |          |          |
|---|---|---|
| ![GIF](readme_images/sample_0.gif) | ![GIF](readme_images/sample_1.gif) | ![GIF](readme_images/sample_2.gif) |
| ![GIF](readme_images/sample_3.gif) | ![GIF](readme_images/sample_4.gif) | ![GIF](readme_images/sample_5.gif) |
| ![GIF](readme_images/sample_6.gif) | ![GIF](readme_images/sample_7.gif) | ![GIF](readme_images/sample_8.gif) |
| ![GIF](readme_images/sample_9.gif) | ![GIF](readme_images/sample_10.gif) | ![GIF](readme_images/sample_11.gif) |
| ![GIF](readme_images/sample_12.gif) | ![GIF](readme_images/sample_13.gif) | ![GIF](readme_images/sample_14.gif) |
| ![GIF](readme_images/sample_15.gif) | ![GIF](readme_images/sample_16.gif) | ![GIF](readme_images/sample_17.gif) |
