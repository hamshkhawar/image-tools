# Image Analysis Workflow
This repository contains jupyter notebook workflows for high throughput image processing and analysis using Polus-Plugins API and polus-data on jupyter hub.

## Pre-requisites

#### 1.

Set up the [AWS CLI](https://aws.amazon.com/cli/) and your [S3 keys](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

#### 2.
* If Jupyter hub is used to execute these workflow so in that case first create a new conda environment to install the required packages.
* mkdir -p ~/work/my-conda-envs/py39
* mamba env create --prefix work/my-conda-envs/py39 --file path-to/environment.yaml
* mv /home/jovyan/work/my-conda-envs/py39/share/jupyter/kernels/python3 /home/jovyan/work/my-conda-envs/py39/share/jupyter/kernels/py39
* /home/jovyan/work/my-conda-envs/py39/bin/python -m ipykernel install --user --name py39 --display-name "py39"
* source activate /home/jovyan/work/my-conda-envs/py39

## Running the workflow
Currently we have workflow available for followings
* Visualization
   Creates zarr pyramids of and entire 384 well plate images to be visualized with polus viv rendering application
* Segmentation & Nyxus Feature Extraction
   Segmentation of objects using pretrained model from polus-smp-training-plugin and nyxus feature extraction
* Analysis
   Four different thresholding methods used to detect COVID positive cells

In each `notebook`, set any `dry_run` parameter to `False` to run that part of the workflow.
