# Polus Image Analysis Workflow
This repository contains workflows for high throughput image processing and analysis using Polus-Plugins API and polus-data

## Pre-requisites

#### 1.

Set up the [AWS CLI](https://aws.amazon.com/cli/) and your [S3 keys](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

#### 2.
* First create a new conda environment to install the required packages.
* conda create -n myenv
* conda activate myenv
* git clone https://github.com/Nicholas-Schaub/polus-data.git
* git branch -r
* git checkout origin/Eastman2021Infectivity
* Use pip to install [polus-data](https://github.com/LabShare/polus-data.git)
#### 3.
To install polus-plugins as package. First clone a repository
* git clone https://Personal_access_token@github.com/PolusAI/polus-plugins.git
* cd polus-plugin
* git branch -r
* git checkout origin/feat/package
* pip install -r requirements.txt
* pip install -e .
* Type environmental varialble: export GITHUB_AUTH=Personal_access_token 
## Running the workflow
In `main.py`, set any `dry_run` parameter to `False` to run that part of the workflow.


