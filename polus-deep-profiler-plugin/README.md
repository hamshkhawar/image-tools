# DeepProfiler


This plugin extracts deep learning features at the resolution of single cell level from fluorescent microscopy images


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 7 input arguments and
1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inputDir` | Input Intensity images collection to be processed by this plugin | Input | collection |
| `--maskDir` | Input mask images collection to be processed by this plugin | Input | collection |
| `--inputcsv` | Input csv file containing boundingbox coordinates for cells | Input | csvcollection |
| `--model` | Choose a model for Feature Extraction| Input | enum |
| `--batchsize` | Choose a batchsize for cells to be used for model prediction|Input | number|
| `--filename` | Filename of the output CSV | Input | string |
| `--outDir` | Output collection | Output | collection |


