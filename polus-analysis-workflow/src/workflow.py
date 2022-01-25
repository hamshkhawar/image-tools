from pathlib import Path
from queue import Empty
import subprocess
import typing
from typing import List, Union, Optional, Dict
import logging
import os
import re



logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)


class workflow:

    def __init__(self, DATA_DIR:Path,PLUGIN_NAME:str,VERSION:str, TAG:str, OUT_DIR:str):
        self.DATA_DIR = DATA_DIR
        self.PLUGIN_NAME=PLUGIN_NAME
        self.VERSION=PLUGIN_NAME
        self.TAG=TAG
        self.OUT_DIR=OUT_DIR

    def create_output_folder(self):
        outname =self.PLUGIN_NAME.split('-')[1:-1]
        if not outname:
            outname = self.PLUGIN_NAME + '-' + 'outputs'
        else:
            outname = "-".join(outname) + '-' + 'outputs'
        outpath = Path(self.OUT_DIR).joinpath(outname)
        if not outpath.exists():
            os.makedirs(Path(outpath))
            logger.info(f'{outname} directory is created')
        else:
            logger.info(f'{outname} directory already exists')
        return outname

 
    
    def assigning_pathnames(self):
        path = Path(DATA_DIR)
        DATA_DIRNAME = path.name
        ROOT_DIR = path.parent
        TARGET_DIR = '/' + path.parts[-2]
        return DATA_DIRNAME, ROOT_DIR, TARGET_DIR
   
    def pull_docker_image(self):
        command = ['docker', 'pull', f'{self.TAG}/{self.PLUGIN_NAME}:{self.VERSION}'] 
        command = " ".join(command)
        os.system(command)
        logger.info(f'{self.PLUGIN_NAME} Pulling docker image')
    
    @staticmethod
    def run_command(PLUGIN_NAME:str, 
                    ROOT_DIR:Path, 
                    TARGET_DIR:Path,
                    TAG:str, 
                    VERSION:str, 
                    ARGS:Dict[str, str]
                )-> None:  
        command = [
                    'docker',
                    'run',
                    '-v',
                    f'{ROOT_DIR}:{TARGET_DIR}',
                    f'{TAG}/{PLUGIN_NAME}:{VERSION}',
                ]
        command.extend(f'--{i}={o}' for i, o in ARGS.items())

        if PLUGIN_NAME == 'nyxus':
            command = ' '.join(command)
            os.system(command)
        else:
            p = subprocess.Popen(command, stdout=subprocess.PIPE)
            stdout, sterr = p.communicate()

def assigning_pathnames(dirpath:Path):
    path = Path(dirpath)
    DATA_DIRNAME = path.name
    ROOT_DIR = path.parent
    TARGET_DIR = '/' + path.parts[-2]
    return DATA_DIRNAME, ROOT_DIR, TARGET_DIR
        
    
def Run_FlatField_Correction(DATA_DIR:Path,
                            FILEPATTREN:str,
                            OUT_DIR:Path,
                            VERSION:Optional[str] = None,
                            dryrun:bool=True)-> None:
    VERSION='1.2.8'  
    PLUGIN_NAME='polus-basic-flatfield-correction-plugin'
    TAG='labshare'
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG, OUT_DIR)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    outname=w.create_output_folder()
    inpDir=Path(TARGET_DIR, DATA_DIRNAME)
    filePattern = FILEPATTREN
    darkfield = 'true'
    photobleach = 'false'
    groupBy= 'xytp'
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
    'inpDir': inpDir,
    'filePattern': filePattern,
    'darkfield': darkfield,
    'photobleach': photobleach,
    'groupBy': groupBy,
    'outDir': outDir 
}
    
    w.pull_docker_image()
    
    if dryrun:
        w.run_command(PLUGIN_NAME,
                    ROOT_DIR,
                    TARGET_DIR,
                    TAG,
                    VERSION,
                    ARGS)




      
# FILEPATTREN = 'p00_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
# DATA_DIR = '/home/ec2-user/data/inputs'
# Run_FlatField_Correction(DATA_DIR, FILEPATTREN)


def Apply_FlatField_Correction(DATA_DIR:Path,  FILEPATTREN:str, OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):

    PLUGIN_NAME='polus-apply-flatfield-plugin'
    VERSION='1.0.6'
    TAG='labshare'
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG, OUT_DIR)
    outname=w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    FF_DIR = Path(TARGET_DIR, 'basic-flatfield-correction-outputs', 'images')
    darkPattern='p00_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_darkfield.ome.tif'
    ffDir=FF_DIR
    brightPattern='p00_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_flatfield.ome.tif'
    imgDir=Path(TARGET_DIR, DATA_DIRNAME)
    imgPattern=FILEPATTREN
    #photoPattern=''

    # # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'imgDir': imgDir,
            'imgPattern': imgPattern,
            'ffDir': ffDir,
            'brightPattern': brightPattern,
            'outDir': outDir,
            'darkPattern': darkPattern 
    }
    w.pull_docker_image()
    if dryrun:
        w.run_command(PLUGIN_NAME,
                    ROOT_DIR,
                    TARGET_DIR,
                    TAG,
                    VERSION,
                    ARGS)



# FILEPATTREN = 'x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
# DATA_DIR = '/home/ec2-user/data/inputs'
# FF_DIR='/home/ec2-user/data/basic-flatfield-correction-outputs/images'
# Apply_FlatField_Correction(DATA_DIR,FF_DIR, FILEPATTREN)



def Run_Montage(DATA_DIR:Path, OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='polus-montage-plugin'
    VERSION='0.3.0'
    TAG='labshare' 
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    outname=w.create_output_folder()

    inpDir=Path(TARGET_DIR, DATA_DIRNAME)
    filePattern='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
    layout='tp,xy'
    imageSpacing='1'
    gridSpacing='20'
    flipAxis='p'

    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'filePattern': filePattern,
            'inpDir': inpDir,
            'flipAxis': flipAxis,
            'layout': layout,
            'outDir': outDir
    }
    w.pull_docker_image()
    if dryrun:
        w.run_command(PLUGIN_NAME,
                    ROOT_DIR,
                    TARGET_DIR,
                    TAG,
                    VERSION,
                    ARGS)

# DATA_DIR='/home/ec2-user/data/apply-flatfield-outputs'
# Run_Montage(DATA_DIR,OUT_DIR)


def Run_Recycle_Vector(DATA_DIR:Path, STICH_DIR:Path,  OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='polus-recycle-vector-plugin'
    VERSION='1.4.3'
    TAG='labshare' 
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    STICHDIR_NAME, _, _ = assigning_pathnames(STICH_DIR)
    outname=w.create_output_folder()
    stitchRegex='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
    collectionRegex='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
    stitchDir=Path(TARGET_DIR, STICHDIR_NAME)
    collectionDir=Path(TARGET_DIR, DATA_DIRNAME)
    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'stitchRegex': stitchRegex,
            'collectionRegex': collectionRegex,
            'stitchDir': stitchDir,
            'collectionDir': collectionDir,
            'outDir': outDir
    }

    w.pull_docker_image()
    if dryrun:
        w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
# DATA_DIR = '/home/ec2-user/data/apply-flatfield-outputs'
# STICH_DIR = '/home/ec2-user/data/montage-outputs'
# Run_Recycle_Vector(DATA_DIR, STICH_DIR)



def Run_Image_Assembler(DATA_DIR:Path, STICH_DIR:Path,  OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='polus-image-assembler-plugin'
    VERSION='1.1.2'
    TAG='labshare'
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG, OUT_DIR) 
    outname=w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    STICHDIR_NAME, _, _ = assigning_pathnames(STICH_DIR)
  
    stitchPath=Path(TARGET_DIR, STICHDIR_NAME)
    imgPath=Path(TARGET_DIR, DATA_DIRNAME)
    timesliceNaming='false'
    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'stitchPath': stitchPath,
            'imgPath': imgPath,
            'outDir': outDir,
            'timesliceNaming': timesliceNaming
           
    }
    w.pull_docker_image()
    if dryrun:
        w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
  

# DATA_DIR = '/home/ec2-user/data/apply-flatfield-outputs'
# STICH_DIR = '/home/ec2-user/data/recycle-vector-outputs'
# Run_Image_Assembler(DATA_DIR, STICH_DIR)


def Run_precompute_slide(DATA_DIR:Path, IMAGETYPE:str, OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='polus-precompute-slide-plugin'
    VERSION='1.3.12'
    TAG='labshare' 
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR) 
    outname = w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)

    inpDir=Path(TARGET_DIR, DATA_DIRNAME)
    pyramidType='Neuroglancer'
    filePattern='x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}.ome.tif'
    imageType=IMAGETYPE
    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'inpDir': inpDir,
            'pyramidType': pyramidType,
            'filePattern': filePattern,
            'imageType': imageType,
            'outDir':outDir
           
    }
    w.pull_docker_image()
    if dryrun:
            w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)

# DATA_DIR = '/home/ec2-user/data/image-assembler-outputs'
# IMAGETYPE='image'
# Run_precompute_slide(DATA_DIR, IMAGETYPE)



def Run_SplineDist(DATA_DIR:Path, MODEL_DIR:Path, OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='polus-splinedist-inference-plugin'
    VERSION='eastman03'
    TAG='labshare'    
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR) 
    outname = w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    MODEL_DIRNAME, _, _ = assigning_pathnames(MODEL_DIR)

    inpImageDir=Path(TARGET_DIR, DATA_DIRNAME)
    inpBaseDir=Path(TARGET_DIR, MODEL_DIRNAME)
    imagePattern='x{x+}_y{y+}_wx{t}_wy{r}_c{c}.ome.tif'
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'inpImageDir': inpImageDir,
            'inpBaseDir': inpBaseDir,
            'imagePattern':imagePattern,
            'outDir': outDir
    }

    w.create_output_folder()
    w.pull_docker_image()
    if dryrun:
            w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
# DATA_DIR = '/home/ec2-user/data/apply-outputs'
# Run_SplineDist(DATA_DIR, MODEL_DIRNAME='model')


def Run_Nyxus(DATA_DIR:Path, SEG_DIR:Path, FEATURES_TYPE:str, OUT_DIR:Path, VERSION:Optional[str] = None, dryrun:bool=True):
    PLUGIN_NAME='nyxus'
    VERSION='0.2.4'
    TAG='polusai'
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR) 
    outname = w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    SEG_DIRNAME, _, _ = assigning_pathnames(SEG_DIR)
    outDir = f'{TARGET_DIR}/{outname}'
    w.create_output_folder()
    
    intDir=Path(TARGET_DIR, DATA_DIRNAME)
    if FEATURES_TYPE == 'Images':
        segDir = intDir
    else:  
        segDir=Path(TARGET_DIR, SEG_DIRNAME)
    filePattern='.*c2\.ome\.tif'
    csvFile='singlecsv'
    features='*ALL_INTENSITY*'
    # features='*BASIC_MORPHOLOGY*'
    outDir = f'{TARGET_DIR}/{outname}'
    

    ARGS = {
            'intDir': intDir,
            'segDir': segDir,
            'outDir': outDir,
            'filePattern': filePattern,
            'csvFile': csvFile,
            'features': features
    }

    w.pull_docker_image()
    if dryrun:
            w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
    

# DATA_DIR='/home/ec2-user/data/images'

# SEG_DIR='/home/ec2-user/data/labels'

# Run_Nyxus(DATA_DIR, SEG_DIR, FEATURES_TYPE = 'labels')


def Run_Imagenet_Model_Featurization(DATA_DIR:Path,
                                    OUT_DIR:Path, 
                                    model:str, 
                                    resolution:str, 
                                    VERSION:Optional[str] = None,
                                    dryrun:bool=True):
    PLUGIN_NAME='polus-imagenet-model-featurization-plugin'
    VERSION='0.1.2'
    TAG='labshare'    
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR) 
    outname = w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    
    inpDir=Path(TARGET_DIR, DATA_DIRNAME)
    outDir = f'{TARGET_DIR}/{outname}'
    
    ARGS = {
            'inpDir': inpDir,
            'outDir':outDir,
            'model': model,
            'resolution': resolution
    }
    w.pull_docker_image()
    if dryrun:
            w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
    
    
# DATA_DIR='/home/ec2-user/data/images'
# Run_Imagenet_Model_Featurization(DATA_DIR, model='VGG19', resolution='500x500')

def Run_DeepProfiler(DATA_DIR:Path,
                    LABEL_DIR:Path,
                    FEAT_DIR:Path,
                    model:str, 
                    batchSize:int,
                    OUT_DIR:Path, 
                    VERSION:Optional[str] = None,
                    dryrun:bool=True):
    PLUGIN_NAME='polus-deep-profiler-plugin'
    VERSION='0.1.6'
    TAG='polusai'    
    w = workflow(DATA_DIR,PLUGIN_NAME,VERSION, TAG,OUT_DIR) 
    outname = w.create_output_folder()
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    LABEL_DIRNAME, _, _ = assigning_pathnames(LABEL_DIR)
    FEAT_DIRNAME, _, _ = assigning_pathnames(FEAT_DIR)
   
    
    inputDir=Path(TARGET_DIR, DATA_DIRNAME)
    maskDir=Path(TARGET_DIR, LABEL_DIRNAME)
    featDir=Path(TARGET_DIR, FEAT_DIRNAME)
    outDir = f'{TARGET_DIR}/{outname}'
    
    ARGS = {
            'inputDir': inputDir,
            'maskDir': maskDir,
            'featDir':featDir,
            'model': model,
            'batchSize': batchSize,
            'outDir': outDir
            
    }

    w.create_output_folder()
    w.pull_docker_image()
    if dryrun:
            w.run_command(PLUGIN_NAME,
                ROOT_DIR,
                TARGET_DIR,
                TAG,
                VERSION,
                ARGS)
    
                    
# DATA_DIR='/home/ec2-user/data/images'
# LABEL_DIR='/home/ec2-user/data/labels'
# FEAT_DIR='/home/ec2-user/data/nyxus-outputs_labels'
# model='VGG16'
# batchSize=8

# Run_DeepProfiler(DATA_DIR,
#                     LABEL_DIR,
#                     FEAT_DIR,
#                     model,
#                     batchSize)