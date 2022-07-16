import os, torch, re, time, subprocess, pathlib, logging
import polus.plugins
from polus.plugins import plugins
from polus.data import collections
from pathlib import Path
from typing import Optional, Dict
import numpy as np



def timer():
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            tic = time.perf_counter()  # more precise than '.clock'
            f(*args, **kwargs)
            toc = time.perf_counter()
            method_name = f.__name__
            logger.info('{}: {:.2f}sec'.format(method_name, toc - tic))
        return wrapped_f
    return wrapper


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)


device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    gpus = None
else:
    gpus = torch.cuda.device_count()
    
logger.info(f'device:{device}, gpus:{gpus}')



def timer():
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            tic = time.perf_counter()  # more precise than '.clock'
            f(*args, **kwargs)
            toc = time.perf_counter()
            method_name = f.__name__
            logger.info('{}: {:.2f}sec'.format(method_name, toc - tic))
        return wrapped_f
    return wrapper


def create_output_folder(outDir, pluginName) -> pathlib.Path:
    if ' ' in pluginName:   
        outname = '_'.join(pluginName.split()) 
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
                '--gpus all',
                '-it',
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

    return command

def Run_FlatField_Correction(inpDir:pathlib.Path,
                            filePattern:str,
                            groupBy:str,
                            outDir:pathlib.Path,
                            dryrun:bool=True) -> None:
    url = 'https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/flatfield/regression/polus-basic-flatfield-correction-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.BasicFlatfieldCorrectionPlugin
    pluginName = pl.name
    pluginName = '_'.join(pluginName.split())
    pl.inpDir = inpDir
    pl.filePattern = filePattern
    pl.darkfield = True
    pl.photobleach=False
    pl.groupBy = groupBy
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    newpath = Path(outpath, 'images')
    return newpath


def ApplyFlatfield(inpDir:pathlib.Path,
                            filePattern:str,
                            plate:str,
                            outDir:pathlib.Path,
                            ffDir:Optional[Path]=None,
                            dryrun:bool=True) -> None:
    url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/flatfield/transforms/images/polus-apply-flatfield-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.ApplyFlatfield
    pluginName = pl.name
    pluginName = '_'.join(pluginName.split())
    pl.imgDir = inpDir
    pl.imgPattern = filePattern
    pl.ffDir= ffDir
    pl.brightPattern='p'+ plate + '_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_flatfield.ome.tif'
    pl.darkPattern='p'+ plate + '_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_darkfield.ome.tif'
    pl.photoPattern = ''
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def Run_Montage(inpDir:pathlib.Path,
                filePattern:str,
                outDir:pathlib.Path,
                dryrun:bool=True
                ) -> None:
    url='https://raw.githubusercontent.com/Nicholas-Schaub/polus-plugins/fix/montage/transforms/images/polus-montage-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.Montage
    pluginName = pl.name
    pl.inpDir=inpDir
    pl.filePattern=filePattern
    pl.layout = 'tp, xy'
    pl.imageSpacing=1
    pl.gridSpacing=20
    pl.flipAxis='p'
    outpath, _ = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def Recycle_Vector(inpDir:pathlib.Path, stitchDir:pathlib.Path, filePattern:str, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url = Path('/home/ec2-user/Anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/labshare/Recycle.json')
    # url = 'https://raw.githubusercontent.com/PolusAI/polus-plugins/master/transforms/polus-recycle-vector-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.RecycleStitchingVectorPlugin1
    pluginName = pl.name
    # pl.filepattern=filePattern
    pl.stitchDir=stitchDir
    pl.collectionDir=inpDir
    pl.stitchRegex=filePattern
    pl.collectionRegex=filePattern
    pl.groupBy='xytp'
    outpath, _ = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def Image_Assembler(inpDir:pathlib.Path, stitchPath:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url='https://raw.githubusercontent.com/PolusAI/polus-plugins/dev/transforms/images/polus-image-assembler-plugin/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.ImageAssembler
    pluginName = pl.name
    pl.stitchPath=stitchPath
    pl.imgPath=inpDir
    pl.timesliceNaming='false'
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def precompute_slide(inpDir:pathlib.Path, plate:str, pyramidType: str, imageType:str, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    #url='https://raw.githubusercontent.com/PolusAI/polus-plugins/master/visualization/polus-precompute-slide-plugin/plugin.json'
    # url = Path('/home/ec2-user/Anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/labshare/PolusPrecomputeSlidePlugin_M1m4p3.json')
    filePattern= 'p'+plate+'_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}.ome.tif'
    # polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.PolusPrecomputeSlidePlugin1
    pluginName = pl.name
    pl.inpDir = inpDir
    pl.pyramidType=pyramidType
    pl.filePattern=filePattern
    pl.imageType=imageType
    # Output pathsls
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def SplineDist(inpDir:pathlib.Path, filePattern:str, modelDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    url = Path('/home/ec2-user/data/manifests/splinedist.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.SplinedistInferencePlugin
    pluginName = pl.name
    pl.inpImageDir=inpDir
    pl.inpBaseDir=modelDir
    pl.imagePattern=filePattern
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
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
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
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
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath

def polus_smp_training_inference(inpDir:pathlib.Path, filePattern:str, modelDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    # url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/plugin/smp-training/segmentation/polus-smp-training-plugin/plugin.json'
    url = pathlib.Path('/home/ec2-user/Anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/labshare/DemoSmpTraining_Inference_M0m5p6.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.DemoSmpTraining_Inference
    pluginName = 'smp_training_inference'
    pl.inferenceMode = "active"
    pl.imagesInferenceDir = inpDir
    pl.inferencePattern = filePattern
    pl.pretrainedModel = modelDir
    pl.batchSize=10
    pl.device = "gpu"
    pl.lossName='MCCLoss'
    pl.checkpointFrequency='10'
    pl.maxEpochs='10'
    pl.patience='10'
    pl.minDelta='10'
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outputDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath


def SMP_training_inference(inpDir:pathlib.Path, filePattern:str, modelDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    pluginName="polus-smp-training-plugin"
    version= '0.5.6'
    tag='labshare'
    pull_docker_image(pluginName, version, tag)
    outpath, outname=create_output_folder(outDir, pluginName)
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

    return outpath

def FtlLabel(inpDir:pathlib.Path, outDir:pathlib.Path, VERSION:Optional[str] = None, dryrun:bool=True):
    # url = 'https://raw.githubusercontent.com/nishaq503/polus-plugins/ftl/feat/binarization-threshold/transforms/images/polus-ftl-label-plugin/plugin.json'
    url = Path('/home/ec2-user/Anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/labshare/FtlLabel_M0m3p11.json')
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.FtlLabel
    pluginName = pl.name
    pluginName = '_'.join(pluginName.split())
    pl.inpDir = inpDir
    pl.connectivity = '1'
    pl.binarizationThreshold='0.5'
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=gpus)
    return outpath 

def Nyxus(inpDir:pathlib.Path, segDir:pathlib.Path, filePattern:str, outDir:pathlib.Path,dryrun:bool=True):
    url = pathlib.Path('/home/ec2-user/Anaconda3/envs/py39/lib/python3.9/site-packages/polus/manifests/polusai/nyxus.json')
    # url = 'https://raw.githubusercontent.com/friskluft/nyxus/main/plugin.json'
    polus.plugins.submit_plugin(url, refresh=True)
    pl = plugins.Nyxus
    pluginName = pl.name
    pl.intDir = inpDir
    pl.segDir = segDir
    pl.filePattern = filePattern
    pl.csvFile="separatecsv"
    # pl.features = "*ALL_INTENSITY*,AREA_PIXELS_COUNT"
    pl.features = "INTEGRATED_INTENSITY,MEAN,AREA_PIXELS_COUNT"
    pl.pixelDistance=5
    pl.pixelsPerCentimeter=8361.2
    outpath, outname = create_output_folder(outDir, pluginName)
    pl.outDir=outpath
    if not dryrun:
        pl.run(gpus=None)
    return outpath


def Nyxus_exe(inpDir:pathlib.Path, segDir:pathlib.Path, filePattern:str, outDir:pathlib.Path, dryrun:bool=True) -> pathlib.Path:
    pluginName='nyxus'
    outDir, outname=create_output_folder(outDir, pluginName)
    filePattern=filePattern
    # features="*all*"
    #features="*ALL_INTENSITY*,*ALL_MORPHOLOGY*"
    features="*ALL_INTENSITY*,AREA_PIXELS_COUNT"
    # features="INTEGRATED_INTENSITY,MEAN,AREA_PIXELS_COUNT"
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
    return outDir

def rename_files(inpDir:pathlib.Path, dryrun:bool=True):
    if not dryrun:  
        for files in os.listdir(inpDir):
            if files.endswith('c1.ome.tif'):
                replace_name = files[:-9] + '2.ome.tif'
                os.rename(pathlib.Path(inpDir, files), pathlib.Path(inpDir, replace_name))
    return inpDir


def stichingvector(inpDir:pathlib.Path,
                    outDir:pathlib.Path):
    
    """Create separate stiching vectors txt file for each channel
    Args:
        inpDir (Path): Path to input stiching vector.
        outDir (Path): Path to output folder.
    Returns:
        paths for directory containing separate stiching vectors files for each channel
    """
    
    outname = 'stichingvector'
    outpath = Path(outDir, outname)
    if not outpath.exists():
        os.makedirs(Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists' 
    
    with open(f'{inpDir}/img-global-positions-1.txt','r') as file:
        filedata = file.read()
        file_ch1 = filedata.replace('c2', 'c1')
        file_ch2 = filedata.replace('c1', 'c2')
    with open(f'{outpath}/img-global-positions-1.txt','w') as file:
        file.write(file_ch1)   
    with open(f'{outpath}/img-global-positions-2.txt','w') as file:
        file.write(file_ch2)
        
    return outpath



def separating_stichingvector_channels(inpDir:pathlib.Path,
                                      outDir:pathlib.Path):
    
        
    """Create separate stiching vectors txt file for each channel
    Args:
        inpDir (Path): Path to input stiching vector.
        outDir (Path): Path to output folder.
    Returns:
        paths for directory containing separate stiching vectors files for each channel
    """  

    outname = 'separating_stichingvector_channels'
    outpath = Path(outDir, outname)
    if not outpath.exists():
        os.makedirs(Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists'
        
    vectors = [
            v
            for v in Path(inpDir).iterdir()
            if pathlib.Path(v).name.endswith(".txt")
        ]

        
    for v in vectors:     
        with open(v) as f:
            pattern = re.compile(r'[^.+]c([\d])')
            lines = f.readlines()

            channel_list = [re.findall(pattern, line) for line in lines]

            unique_channels = list(np.unique([list(t) for t in zip(*channel_list)]))

            for ch in unique_channels:

                parsed_channels = [line for line in lines if 'c'+ ch in line]

                with open(f'{outpath}/img-global-positions-{ch}.txt', 'w') as f:
                    for line in parsed_channels:  
                        f.write("%s" % line)

    return outpath
