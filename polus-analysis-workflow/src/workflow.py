import polus.plugins
from polus.plugins import plugins
from polus.data import collections
from pathlib import Path
import os
from typing import Optional
import pathlib, logging


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)



def create_output_folder(outDir, pluginName) -> pathlib.Path:
    outname =  "-".join(pluginName.split()) +'_' +'outputs'
    outpath = Path(outDir, outname)
    if not outpath.exists():
        os.makedirs(Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists'
    return outpath


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

def SMP_training_inference(inpDir:pathlib.Path, filePattern:str, model:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/plugin/smp-training/segmentation/polus-smp-training-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.DemoSMPTrainingInference
    pluginName = pl.name
    pl.inferenceMode = "active"
    pl.imagesInferenceDir = inpDir
    pl.inferencePattern = filePattern
    pl.inferenceMode=True
    pl.pretrainedModel = model
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath


def FtlLabel(inpDir:pathlib.Path, outDir:pathlib.Path, connectivity:Optional[int] = 1, binarizationThreshold:Optional[int] = 0.5, VERSION:Optional[str] = None, dryrun:bool=True):
    url = 'https://raw.githubusercontent.com/PolusAI/polus-plugins/master/transforms/images/polus-ftl-label-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.FtlLabel
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.connectivity = connectivity
    pl.binarizationThreshold=binarizationThreshold
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath

def Nyxus(inpDir:pathlib.Path, segDir:pathlib.Path, filePattern:str, csvFile:str, outDir:pathlib.Path, features:Optional[str] = "*ALL*", VERSION:Optional[str] = None, dryrun:bool=True):
    url = 'https://raw.githubusercontent.com/friskluft/nyxus/main/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.Nyxus
    pluginName = pl.name
    pl.intDir = inpDir
    pl.segDir = segDir
    pl.filePattern = ".*"
    pl.csvFile=csvFile
    pl.features = features
    outpath = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath


