import polus.plugins
from polus.plugins import plugins
from polus.data import collections
from pathlib import Path
import os
from typing import Optional, Dict
import pathlib, logging
import subprocess


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)


def create_output_folder(outDir, pluginName) -> pathlib.Path:
    if 'plugin' in pluginName:
        outname = '_'.join(pluginName.split('-')[:-1]) 
        outname = f'{outname}_outputs'
    else:
        outname = f'{pluginName}_outputs'
    outpath = Path(outDir, outname)
    if not outpath.exists():
        os.makedirs(Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists'
    return outpath, outname

def pull_docker_image(plugin_name:str, 
                    version:str, tag:str)-> None:
    command = ['docker', 'pull', f'{tag}/{plugin_name}:{version}'] 
    command = " ".join(command)
    os.system(command)
    print(f'{plugin_name} Pulling docker image')

def creating_volume(volume_name:str):
    command = ['docker','volume','create',f'{volume_name}']
    command = ' '.join(command)
    os.system(command)
    f'Creating docker volume locally: {volume_name}'
    return

def removing_volume(volume_name:str):
    command = ['docker','volume', 'rm',f'{volume_name}']
    command = ' '.join(command)
    f'Removing docker volume locally: {volume_name}'
    return os.system(command)

def run_command(pluginName:str, 
                root_dir:pathlib.Path, 
                volume_name:str,
                tag:str, 
                version:str, 
                args:Dict[str, str]
                )-> None:
    command = [
                'docker',
                'run',
                '-v',
                f'{root_dir}:/{volume_name}',
                f'{tag}/{pluginName}:{version}'
            ]
    command.extend(f'--{i}={o}' for i, o in args.items())
    if pluginName == 'nyxus':
        command = ' '.join(command)
        os.system(command)
    else:
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        stdout, sterr = p.communicate()

    return

def Run_FlatField_Correction(inpDir:pathlib.Path,
                            filePattern:str,
                            groupBy:str,
                            outDir:pathlib.Path,
                            dryrun:bool=True) -> None:
    url = 'https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/flatfield/regression/polus-basic-flatfield-correction-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.BasicFlatfieldCorrectionPlugin
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.filePattern = filePattern
    pl.darkfield = True
    pl.photobleach=False
    pl.groupBy = groupBy
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    newpath = Path(outpath, 'images')
    return newpath

def ApplyFlatfield(inpDir:pathlib.Path,
                            filePattern:str,
                            outDir:pathlib.Path,
                            ffDir:Optional[Path]=None,
                            dryrun:bool=True) -> None:
    url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/flatfield/transforms/images/polus-apply-flatfield-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.ApplyFlatfield
    pluginName = pl.name
    pl.imgDir = inpDir
    pl.imgPattern = filePattern
    pl.ffDir= ffDir
    pl.brightPattern = 'p01_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_flatfield.ome.tif'
    pl.darkPattern = 'p01_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_darkfield.ome.tif'
    pl.photoPattern = ''
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def Run_Montage(inpDir:pathlib.Path,
                filePattern:str,
                outDir:pathlib.Path,
                dryrun:bool=True
                ) -> None:
    # url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/montage/transforms/images/polus-montage-plugin/plugin.json'
    url=Path('/home/ec2-user/data/plugin-manifest/montage.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.Montage2
    pluginName = pl.name
    pl.inpDir=inpDir
    pl.filePattern=filePattern
    pl.layout = 'tp, xy'
    pl.imageSpacing=1
    pl.gridSpacing=20
    pl.flipAxis='p'
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def Recycle_Vector(inpDir:pathlib.Path, stitchDir:pathlib.Path, filePattern:str, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    pl = plugins.RecycleStitchingVectorPlugin
    pluginName = pl.name
    pl.filepattern=filePattern
    pl.stitchDir=stitchDir
    pl.collectionDir=inpDir
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def Image_Assembler(inpDir:pathlib.Path, stitchPath:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url='https://raw.githubusercontent.com/PolusAI/polus-plugins/dev/transforms/images/polus-image-assembler-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.ImageAssembler
    pluginName = pl.name
    pl.stitchPath=stitchPath
    pl.imgPath=inpDir
    pl.timesliceNaming='false'
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def precompute_slide(inpDir:pathlib.Path, filePattern:str, imageType:str, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    pl = plugins.PolusPrecomputeSlidePlugin
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.pyramidType='Neuroglancer'
    pl.filePattern=filePattern
    pl.imageType=imageType
    # Output paths
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def SplineDist(inpDir:pathlib.Path, filePattern:str, modelDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url = Path('/home/ec2-user/data/manifests/splinedist.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.SplinedistInferencePlugin
    pluginName = pl.name
    pl.inpImageDir=inpDir
    pl.inpBaseDir=modelDir
    pl.imagePattern=filePattern
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def ImagenetModelFeaturization(inpDir:pathlib.Path, model:str, resolution:str, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    models = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
     'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 
     'InceptionResNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201']
    assert model in models   
    pl = plugins.ImagenetModelFeaturization
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.model = model
    pl.resolution = resolution
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath


def cellposeInference(inpDir:pathlib.Path, filePattern:str,outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/plugin/cellpose-inference/segmentation/polus-cellpose-inference-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.CellposeInference
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.filePattern = filePattern
    pl.diameter=0
    pl.diameterMode='PixelSize'
    pl.pretrainedModel = 'nuclei'
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

# def SMP_training_inference(inpDir:pathlib.Path, filePattern:str, model:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
#     url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/plugin/smp-training/segmentation/polus-smp-training-plugin/plugin.json'
#     polus.plugins.submit_plugin(url, refresh=True)
#     pl = plugins.DemoSMPTrainingInference
#     pluginName = pl.name
#     pl.inferenceMode = "active"
#     pl.imagesInferenceDir = inpDir
#     pl.inferencePattern = filePattern
#     pl.inferenceMode=True
#     pl.pretrainedModel = model
#     outpath = create_output_folder(outDir, pluginName)
#     pl.outDir=outpath
#     if not dryrun:
#         pl.run(gpus=None)
#     return outpath


def SMP_training_inference(inpDir:pathlib.Path, filePattern:str, modelDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    pluginName="polus-smp-training-plugin"
    version= '0.5.6'
    tag='labshare'
    pull_docker_image(pluginName, version, tag)
    _, outname=create_output_folder(outDir, pluginName)
    root_dir=os.path.split(inpDir)[0]
    inpDir = os.path.split(inpDir)[-1]
    model_dir = os.path.split(modelDir)[-1]
    volume_name='data2'
    creating_volume(volume_name)
    args = {
            'inferenceMode': 'active',
            'imagesInferenceDir': f'/{volume_name}/{inpDir}',
            'inferencePattern': filePattern,
            'pretrainedModel': f'/{volume_name}/{model_dir}',
            'outputDir': f'/{volume_name}/{outname}'}

    if not dryrun:
        run_command(pluginName, root_dir, volume_name, tag, version, args)

    print(f'Removing docker volume after finished running docker image')

    removing_volume(volume_name)

    return 

def FtlLabel(inpDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    # url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/ftl/feat/binarization-threshold/transforms/images/polus-ftl-label-plugin/plugin.json'
    url = Path('/home/ec2-user/anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/labshare/FtlLabel_M0m3p11.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.FtlLabel
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.connectivity = '1'
    pl.binarizationThreshold='0.5'
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath 

# def Nyxus(inpDir:pathlib.Path, segDir:pathlib.Path, filePattern:str, csvFile:str, outDir:pathlib.Path, features:Optional[str] = "*ALL*", VERSION:Optional[str] = None, dryrun:bool=True):
#     url = 'https://raw.githubusercontent.com/friskluft/nyxus/main/plugin.json'
#     polus.plugins.submit_plugin(url, refresh=True)
#     pl = plugins.Nyxus
#     pluginName = pl.name
#     pl.intDir = inpDir
#     pl.segDir = segDir
#     pl.filePattern = ".*"
#     pl.csvFile=csvFile
#     pl.features = features
#     outpath = create_output_folder(outDir, pluginName)
#     pl.outDir=outpath
#     if not dryrun:
#         pl.run(gpus=None)
#     return outpath


def Nyxus_exe(inpDir:pathlib.Path, segDir:pathlib.Path, filePattern:str, outDir:pathlib.Path, dryrun:bool=True):
    pluginName='nyxus'
    outDir, outname=create_output_folder(outDir, pluginName)
    filePattern='.*'
    features="*all*"
    csvFile="separatecsv"
    ARGS = {
        'intDir': inpDir,
        'segDir': segDir,
        'outDir': outDir,
        'filePattern': filePattern,
        'csvFile': csvFile,
        'features': features
    }
    command = [
                '/home/ec2-user/nyxus-20220520.linuxexe'             
            ]
    command.extend(f'--{i}={o}' for i, o in ARGS.items())
    command = ' '.join(command)
    if not dryrun:
        os.system(command)
    return 


