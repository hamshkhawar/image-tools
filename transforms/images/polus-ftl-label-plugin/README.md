# FTL Label

This plugin is an n-dimensional connected component algorithm that is similar to the [Light Speed Labeling](http://www-soc.lip6.fr/~lacas/Publications/ICIP09_LSL.pdf) algorithm.
This algorithm works in n-dimensions and uses run length encoding to compress the image and accelerate computation.
As a reference to the Light Speed Labeling algorithm, we named this method the Faster Than Light (FTL) Labeling algorithm, although this should not be interpreted as this algorithm being faster.

The `Cython` implementation generally performs better than [SciKit's `label` method](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label), except on images with random noise in which it performs up to 1more0x slower and uses 2x more memory.
In most of our real test images, this algorithm ran 2x faster and used 4x less memory.
This implementation does load the entire image into memory and, so, is not suitable for extremely large images.

The `Rust` implementation processes the images in tiles rather than all at once.
This lets it scale to arbitrarily large sizes but does make it slower than the Cython implementation.
However, most of the bottleneck is in the interface between `Python` and `Rust`.
The Rust implementation works with 2d and 3d images.

To see detailed documentation for the `Rust` implementation you need to:
 * Install [Rust](https://doc.rust-lang.org/stable/book/ch01-01-installation.html),
 * add Cargo to your `PATH`, and
 * run from the terminal (in this directory): `cargo doc --open`.

That last command will generate documentation and open a new tab in your default web browser.

We determine whether to use the `Cython` or `Rust` implementation on a per-image basis depending on the size of that image.
If we expect the image to occupy less than `500MB` of memory, we use the `Cython` implementation otherwise we use the `Rust` implementation. 

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## To do

The following optimizations should be added to increase the speed or decrease the memory used by the plugin.
1. Implement existing specialized C++ methods that accelerate the run length encoding operation by a factor of 5-10

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name             | Description                                           | I/O    | Type       |
|------------------|-------------------------------------------------------|--------|------------|
| `--inpDir`       | Input image collection to be processed by this plugin | Input  | collection |
| `--connectivity` | City block connectivity                               | Input  | number     |
| `--outDir`       | Output collection                                     | Output | collection |

**NOTE:**
Connectivity uses [SciKit's](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label) notation for connectedness, which we call cityblock notation.
For 2D, 1-connectivity is the same as 4-connectivity and in 3D is the same as 6-connectivity.
As you increase the connectivity, you increase the number of pixel jumps away from the center point.
Each new jump must be orthogonal to all previous jumps.
This means that `connectivity` should have a minimum value of `1` and a maximum value equal to the dimensionality of the images.

SciKit's documentation has a good illustration for 2D:
```
1-connectivity     2-connectivity     diagonal connection close-up

     [ ]           [ ]  [ ]  [ ]             [ ]
      |               \  |  /                 |  <- hop 2
[ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
      |               /  |  \             hop 1
     [ ]           [ ]  [ ]  [ ]
```