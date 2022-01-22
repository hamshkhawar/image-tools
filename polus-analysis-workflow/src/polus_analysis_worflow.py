from argparse import OPTIONAL
from multiprocessing import cpu_count
from queue import Empty
import polus
import polus.plugins as pp
from polus.plugins import plugins
import requests
import json
from urllib import request
import typing
from typing import List, Union, Optional, Dict
from pathlib import Path
import os
import logging
from polus.plugins import Plugin
import string


# polus.plugins.update_polus_plugins()  

# plugins.refresh()


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)

class workflow:

    def __init__(self, Data_Dir:Path,Plugin_Name:str):
        self.Data_Dir = Data_Dir
        self.Plugin_Name=Plugin_Name


    def create_output_folder(self):
        outname =self.Plugin_Name.split('-')[1:-1]
        if not outname:
            outname = self.Plugin_Name + '-' + 'outputs'
        else:
            outname = "-".join(outname) + '-' + 'outputs'
        outpath = Path(Path(Data_Dir).parent, outname)
        if not outpath.exists():
            os.makedirs(Path(outpath))
            logger.info(f'{outname} directory is created')
        else:
            logger.info(f'{outname} directory already exists')
        return outpath


def import_plugin(
    Plugin:str,
    User:str,
    Branch:str,
    Repo:str='polus-plugins',
    Manifest:str='plugin.json',
    ):
    
    url=['https://raw.githubusercontent.com',User,Repo, Branch, Plugin, Manifest]

    url = '/'.join(url)
    plugin = pp.submit_plugin(url)
    pp.plugins.refresh()

    return plugin


def import_plugin_url(
    Url:str):
    plugin = pp.submit_plugin(Url)
    pp.plugins.refresh()
    return plugin



def Run_FlatField_Correction(Data_Dir:Path,
                            FilePattren:str,
                            dryrun:bool=True) -> None:
    Plugin_Name='polus-basic-flatfield-correction-plugin'
    import_plugin(Plugin_Name, 
    'Nicholas-Schaub',
    'fix/flatfield/regression')
    w = workflow(Data_Dir, Plugin_Name)
    plugin = pp.plugins.BasicFlatfieldCorrectionPlugin
    plugin.inpDir = Data_Dir
    plugin.filePattern = FilePattren
    plugin.darkfield = True
    plugin.photobleach=False
    plugin.groupBy = 'xytp'
    plugin.outDir=w.create_output_folder()
    if not dryrun:
        Plugin.run(plugin, gpus=None)

# FilePattren='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
# Data_Dir='/home/ec2-user/data/corrected'

# Run_FlatField_Correction(Data_Dir,
#                             FilePattren,
#                             dryrun=False)


def Apply_FlatField_Correction(Data_Dir:Path,
                            FilePattren:str,
                            FF_Dir:Optional[Path]=None,
                            dryrun:bool=True) -> None:

    if not FF_Dir:
        FF_Dir = Path(Path(Data_Dir).parent, 'basic-flatfield-correction-outputs', 'images')
    
    Plugin_Name='polus-apply-flatfield-plugin'
    url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/flatfield/transforms/images/polus-apply-flatfield-plugin/plugin.json'
    plugin = import_plugin_url(url)
    w = workflow(Data_Dir, Plugin_Name)
    plugin.imgDir = Data_Dir
    plugin.imgPattern=FilePattren
    plugin.ffDir= FF_Dir
    plugin.brightPattern = 'x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_flatfield.ome.tif'
    plugin.outDir=w.create_output_folder()
    plugin.darkPattern = 'x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_darkfield.ome.tif'
    plugin.photoPattern = ' '
    
    
    if not dryrun:
        Plugin.run(plugin, gpus=None)


# Plugin_Name='polus-apply-flatfield-plugin'   
# Data_Dir='/home/ec2-user/data/inputs'
# FilePattren='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
# FF_Dir='/home/ec2-user/data/basic-flatfield-correction-outputs/images'




# Apply_FlatField_Correction(Data_Dir,
#                             FilePattren,
#                             FF_Dir,
#                             dryrun=False)


def Run_Montage(Data_Dir:Path,
                FilePattren:str,
                dryrun:bool=True
                ) -> None:
    Plugin_Name='polus-montage-plugin'
    url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/montage/transforms/images/polus-montage-plugin/plugin.json'
    plugin = import_plugin_url(url)
    plugins.refresh()

    # Plugin_Name='polus-montage-plugin'
    # import_plugin(Plugin_Name, 
    # 'Nicholas-Schaub',
    # 'fix/montage/transforms/images')
    w = workflow(Data_Dir, Plugin_Name)
    plugin = pp.plugins.Montage
    plugin.inpDir=Data_Dir
    plugin.filePattern=FilePattren
    plugin.layout='tp', 'xy'
    plugin.imageSpacing='1'
    plugin.gridSpacing='20'
    plugin.flipAxis='p'

    plugin.outDir=w.create_output_folder()

    if not dryrun:
        Plugin.run(plugin, gpus=None)

    return
Data_Dir='/home/ec2-user/data/corrected'
FilePattren='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
Run_Montage(Data_Dir,FilePattren,dryrun=False)





    