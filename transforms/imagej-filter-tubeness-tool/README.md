# ImageJ filter tubeness v(0.5.1)

This plugin filters a collection to produce a score for how tube-like each point in the image is.

This WIPP plugin was automatically generated by a utility that searches for
ImageJ plugins and generates code to run them. For more information on what this
plugin does, contact one of the authors: Nick Schaub (nick.schaub@nih.gov),
Anjali Taneja or Benjamin Houghton (benjamin.houghton@axleinfo.com).

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

Bump the version in the `VERSION` file.

Then to build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name            | Description                                     | I/O    | Type       |
| --------------- | ----------------------------------------------- | ------ | ---------- |
| `--inpDir`      | Input collection to be processed by this plugin | Input  | collection |
| `--sigma`       | Desired scale in physical units                 | Input  | number     |
| `--calibration` | Physical pixel sizes in all dimensions          | Input  | array      |
| `--outDir`      | Output directory                                | Output | collection |
