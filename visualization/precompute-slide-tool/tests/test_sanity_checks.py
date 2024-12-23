"""Automated sanity checks with multiple inputs."""

from pathlib import Path


def test_precompute(
    plugin_dirs: tuple[Path, Path], random_ome_tiff_images: tuple[Path, str, str],
) -> None:
    """Test the plugin."""
    inp_dir, pyramid_type, image_type = random_ome_tiff_images
    _, out_dir = plugin_dirs

    print(inp_dir)

    # TODO test filepattern

