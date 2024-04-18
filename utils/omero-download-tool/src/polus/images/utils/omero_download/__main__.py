"""Omero Download Tool."""

import logging
from pathlib import Path
from typing import Optional

import polus.images.utils.omero_download.omero_download as od
import typer

app = typer.Typer()

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.images.utils.omero_download")


@app.command()
def main(
    data_type: od.DATATYPE = typer.Option(
        ...,
        "--dataType",
        "-d",
        help="The supported object types to be retreived",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name of the object to be downloaded",
    ),
    object_id: Optional[int] = typer.Option(
        None,
        "--objectId",
        "-i",
        help="Identifier of the object to be downloaded",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Path to directory outputs",
    ),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of files",
    ),
) -> None:
    """Retrieve the microscopy image data from the OMERO NCATS server."""
    logger.info(f"--dataType = {data_type}")
    logger.info(f"--name = {name}")
    logger.info(f"--objectId = {object_id}")
    logger.info(f"--outDir = {out_dir}")

    out_dir = out_dir.resolve()

    if not Path(out_dir).exists():
        out_dir.mkdir(exist_ok=True)

    if not preview:
        model = od.OmeroDwonload(
            data_type=data_type.value,
            name=name,
            object_id=object_id,
            out_dir=out_dir,
        )
        model.get_data()
    else:
        od.generate_preview(out_dir)


if __name__ == "__main__":
    app()