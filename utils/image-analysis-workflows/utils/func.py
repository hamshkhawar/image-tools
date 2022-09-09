import os
import shutil
import pathlib
from pathlib import Path
from typing import Optional, Dict
import re
import numpy as np
import time
import warnings
import subprocess
from functools import partial
from bfio import BioReader, BioWriter
import multiprocessing
import skimage.measure as measure
import shutil
import logging
from functools import wraps


def my_timer(orig_func):
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='runtime.log', level=logging.INFO, filemode='w')

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = round((time.time() - t1) / 60, 4)

        logging.info(f'Functions: {orig_func.__name__} took {t2} minutes')

        return result

    return wrapper



def run_command(pluginName:str, 
                args:Dict[str, str]
                )-> Dict[str, str]:
    
    """Using python subprocess module for executing shell commands
    Args:
        pluginName (str):  Plugin/function name
        args (Dict[str, str]): input arguments
    Returns:
        command
    """
    
    workDir = pathlib.Path(os.getcwd()).parent.parent
    command = [
              'python', f'{workDir}/plugins/{pluginName}/src/main.py'
            ]
    command.extend(f'--{i}={o}' for i, o in args.items())
  
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    stdout, sterr = p.communicate()

    return command



def create_output_folder(outDir:pathlib.Path, 
                         pluginName:str,
                         plate:Optional[str]) -> pathlib.Path:
     
    """Creating output directory of particular plugin/function
    Args:
        outDir (Path): Path to outputs
        pluginName (str):  Plugin/function name
        plate (str): unique regex as identifier for each plate
    Returns:
        paths to outputs
    """
    
    if (len(pluginName) != None) and ('-' in pluginName):
        
        dirname = '_'.join(pluginName.split('-')[1:3])
        outname = f'p{plate}_{dirname}'
    elif (len(pluginName) != None) and not ('-' in pluginName):
        dirname = pluginName
        outname = f'p{plate}_{dirname}'
    else:
        raise ValueError('pluginName is not defined')
    
    if plate:
        outpath = Path(outDir, dirname, outname)
                                        
    else:
        outpath = Path(outDir, dirname)
                                        
    if not outpath.exists():
        os.makedirs(Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists'
    return outpath


@my_timer
def basic_flatfield(inpDir:pathlib.Path,
                   outDir:pathlib.Path,
                   filePattern:str,
                   plate:str,
                   dryrun:bool=True)-> pathlib.Path:
    
    """Run the polus-basic-flatfield-correction-plugin to calculate flatfield information 
    Args:
        inpDir (Path): Path to intensity images
        filePattern (str):  Regex to parse image files
        plate (str): unique regex as identifier for each plate
        outDir (Path): Path to output folder
    Returns:
        paths to darkfield/brightfield and photobleach images
    """
    
    pluginName='polus-basic-flatfield-correction-plugin'
    groupBy='xytp'
    darkfield = True
    photobleach = False

    outpath = create_output_folder(outDir, pluginName, plate)
    args = {
                'inpDir': f'{inpDir}',
                'darkfield': f'{darkfield}',
                'photobleach': f'{photobleach}',
                'filePattern': f'{filePattern}',
                'groupBy': f'{groupBy}',
                'outDir':f'{outpath}'
    }
    
    if not dryrun:       
        run_command(pluginName, args)   
        return outpath


@my_timer    
def apply_flatfield(inpDir:pathlib.Path,
                   outDir:pathlib.Path,
                   ffDir:pathlib.Path,
                   filePattern:str,
                   plate:str,
                   dryrun:bool=True)-> pathlib.Path:
    
    """Run polus-apply-flatfield-plugin to apply flatfield algorithm to a collection of images
    Args:
        inpDir (Path): Path to intensity images
        ffDir (Path): Path to brightfield and/or darkfield images
        filePattern (str):  Regex to parse image files
        plate (str): unique regex as identifier for each plate
        outDir (Path): Path to output folder
    Returns:
        paths for directory containing corrected images
    """
    
    pluginName='polus-apply-flatfield-plugin'
    brightPattern='p'+ plate + "_x(01-24)_y(01-16)_wx(1-3)_wy(1-3)_c{c}_flatfield.ome.tif"
    darkPattern='p'+ plate + "_x(01-24)_y(01-16)_wx(1-3)_wy(1-3)_c{c}_darkfield.ome.tif"
    photoPattern = None
    outpath = create_output_folder(outDir, pluginName, plate=None)

    args = {
                'imgDir': f'{inpDir}',
                'imgPattern': f'{filePattern}',
                'ffDir': f'{ffDir}/images',
                'brightPattern': f'{brightPattern}',
                'outDir': f'{outpath}',
                'darkPattern':f'{darkPattern}'
    }
    
    if not dryrun:
        run_command(pluginName, args)
        
        return outpath
    
@my_timer
def montage(inpDir:pathlib.Path,
                   outDir:pathlib.Path,
                   filePattern:str,
                   plate:str,
                   dryrun:bool=True)-> pathlib.Path:
    
    """Run polus-montage-plugin to build a stitching vector
    Args:
        inpDir (Path): Path to intensity images.
        outDir (Path): Path to output folder.
    Returns:
        paths for directory containing montage stiching vector
    """
    
    pluginName='polus-montage-plugin'
    layout='tp, xy'
    imageSpacing = 1
    gridSpacing = 20
    flipAxis = 'p'
    outpath = create_output_folder(outDir, pluginName, plate)

    args = {
                'filePattern': f'{filePattern}',
                'inpDir': f'{inpDir}',
                'layout': f'{layout}',
                'imageSpacing': f'{imageSpacing}',
                'gridSpacing': f'{gridSpacing}',
                'flipAxis':f'{flipAxis}',
                'outDir':f'{outpath}'

    }
    
    if not dryrun:       
        run_command(pluginName, args)   
        return outpath
    
@my_timer    
def stichingvector(inpDir:pathlib.Path,
                    outDir:pathlib.Path,
                  plate:str,
                  dryrun:bool=True)-> pathlib.Path:
    
    """Create separate stiching vectors txt file for each channel
    Args:
        inpDir (Path): Path to input stiching vector.
        outDir (Path): Path to output folder.
    Returns:
        paths for directory containing separate stiching vectors files for each channel
    """
    dirname = 'stichingvector'
    outname = f'p{plate}_stichingvector'
    outpath = pathlib.Path(outDir, 'outputs', dirname, outname)
    if not outpath.exists():
        os.makedirs(pathlib.Path(outpath))
        f'{outname} directory is created'
    else:
        f'{outname} directory already exists'
        
        
    if not dryrun:

        with open(f'{inpDir}/img-global-positions-1.txt','r') as file:
            filedata = file.read()
            file_ch1 = filedata.replace('c2', 'c1')
            file_ch2 = filedata.replace('c1', 'c2')
        with open(f'{outpath}/img-global-positions-1.txt','w') as file:
            file.write(file_ch1)   
        with open(f'{outpath}/img-global-positions-2.txt','w') as file:
            file.write(file_ch2)
        
    return outpath

@my_timer
def assemble(inpDir:pathlib.Path,
            outDir:pathlib.Path,
            stitchPath:pathlib.Path,
            filePattern:str,
            plate:str,
            dryrun:bool=True)-> pathlib.Path:
    
    """Run polus-image-assembler-plugin to assembles image from a stitching vector
    Args:
        inpDir (Path): Path to intensity image
        outDir (Path): Path to output folder
        stitchPath (Path): Path to stitching vectors
        filePattern (str):  Regex to parse image files
        plate (str): unique regex as identifier for each plate
    Returns:
        paths to assembled images
    """
    
    pluginName='polus-image-assembler-plugin'
#     plate= filePattern.split('_')[0]
    timesliceNaming=True
    outpath = create_output_folder(outDir, pluginName, plate)


    args = {
                'imgPath': f'{inpDir}',
                'stitchPath': f'{stitchPath}',
                'outDir': f'{outpath}',
                'timesliceNaming': f'{timesliceNaming}',


    }

    if not dryrun:       
        run_command(pluginName, args)   
        return outpath
    
    
@my_timer
def precompute_slide(inpDir:pathlib.Path,
                   outDir:pathlib.Path,
                   plate:str,
                   imageType:str,
                   pyramidType:str,
                   dryrun:bool=True)-> pathlib.Path:
    
    """Run polus-precompute-slide-plugin to build full pyramids
    Args:
        inpDir (Path): Path to intensity image
        outDir (Path): Path to output folder
        plate (str): unique regex as identifier for each plate
        imageType (str): Build Zarr, DeepZoom or Neuroglancer pyramid
        pyramidType (str):  Either a image or a segmentation
    Returns:
        paths to full pyramids
    """
    
    pluginName='polus-precompute-slide-plugin'
    outpath = create_output_folder(outDir, pluginName, plate=None)
    filePattern='p'+ plate + "_x(01-24)_y(01-16)_wx(1-3)_wy(1-3)_c{c}.ome.tif"

    args = {
                'inpDir': f'{inpDir}',
                'pyramidType': f'{pyramidType}',
                'filePattern': f'{filePattern}',
                'imageType': f'{imageType}',
                'outDir': f'{outpath}',


    }
    
    if not dryrun:       
        run_command(pluginName, args)   
        return outDir
    
@my_timer    
def convert_label_to_binary(inpDir:pathlib.Path, 
                    outDir:pathlib.Path,
                    pattern:str,
                    dryrun:bool=True)-> pathlib.Path:
    
    """Convert labelled image to binary
    Args:
        inpDir (Path): Path to label image
        outDir (Path): Path to output folder
        plate (str): unique regex as identifier for each plate
    Returns:
        paths to binary images
    """
    
    pluginName='binary'   
    
    outpath = create_output_folder(outDir, pluginName, plate=None)  
    fp = filepattern.FilePattern(inpDir,pattern)

    for f in fp():
        file = f[0]['file'].name
        imgpath = os.path.join(inpDir, file)
        br = BioReader(imgpath)
        img = br.read().squeeze()
        img[img > 0] = 255
        datatype = np.uint8
        with BioWriter(file_path = os.path.join(outpath, file),
                        metadata  = None, 
                        X=img.shape[0],  
                        Y=img.shape[0],
                        dtype = datatype

                        ) as bw:

                bw[:] = img.astype(datatype)
                bw.close()

    return outpath



@my_timer
def SMP_training_inference(inpDir:pathlib.Path, 
                           modelDir:pathlib.Path, 
                           filePattern:str,
                           plate:str,
                           outDir:pathlib.Path,                
                           dryrun:bool=True)-> pathlib.Path:
    """Run polus-smp-training-plugin using pretrained model to predict segmentations
    Args:
        inpDir (Path): Path to intensity images
        modelDir (Path): Path to pretrained model 
        filePattern (str):  Regex to parse image files
        plate (str): unique regex as identifier for each plate
        outDir (Path): Path to output folder
        
    Returns:
        paths to probability images
    """
               
    pluginName="polus-smp-training-plugin"
    outpath = create_output_folder(outDir, pluginName, plate=None)
 
    args = {
            'inferenceMode': 'active',
            'imagesInferenceDir': f'{inpDir}',
            'inferencePattern': f'{filePattern}',
            'pretrainedModel': f'{modelDir}',
            'outputDir': f'{outpath}'}
    
  

    if not dryrun:
        run_command(pluginName, args)
    

    return outpath


def theshoding_probability_image(image:np.ndarray)-> np.ndarray:
    """Threshold probability image
    Args:
        image (np.ndarray): Path to images
        
    Returns:
        Thresholded image
    """
    
    def label(num):

        return 0 if num < 0.5 else 1

    func = np.vectorize(label)
    thresh_img = func(image)
    
    return thresh_img


def labelling_images(outDir:pathlib.Path, flist)-> None:
    
    """Convert image probability to label image
    Args:
        outDir (Path): Path to output folder
        flist  (list): List of image file paths
        
    """ 
    f = os.path.split(flist)[1]
    outname = f.split('.ome.tif')[0] + '.ome.tif'
    br = BioReader(Path(flist))
    img = br.read().squeeze()
    thresh_img = theshoding_probability_image(img)
    label_image = measure.label(thresh_img,return_num=False, connectivity=1)
    datatype = np.uint16
    with BioWriter(file_path = os.path.join(outDir, outname),
                    metadata  = None, 
                    X=img.shape[0],  
                    Y=img.shape[0],
                    dtype = datatype

                    ) as bw:

            bw[:] = label_image.astype(datatype)
            bw.close()

    return

@my_timer
def run_labelling_images(inpDir:pathlib.Path,
                        outDir:pathlib.Path,
                        dryrun:bool=True)-> pathlib.Path:
    
    """Parallel processing of labelling_images function on images
    Args:
        inpDir (Path): Path to images
        outDir (Path): Path to output folder
        plate (str): unique regex as identifier for each plate
        
    Returns:
        paths to labelled images
    """
    pluginName='Labels'
    
    outpath = create_output_folder(outDir, pluginName, plate=None)
        
    flist = [pathlib.Path(inpDir, f) for f in os.listdir(inpDir)]
    num_workers = (multiprocessing.cpu_count() // 2)
    if not dryrun:     
        p = multiprocessing.Pool(processes=num_workers)
        part_func = partial(labelling_images,outpath)
        p.map(part_func, flist)
        p.close()
        p.join()

    return outpath

   
@my_timer    
def rename_files(inpDir:pathlib.Path, dryrun:bool=True)-> pathlib.Path:
    """Renaming channel names in image files
    Args:
        inpDir (Path): Path to images
        
    Returns:
        paths to renamed images
    """
    if not dryrun:  
        for files in os.listdir(inpDir):
            if files.endswith('c1.ome.tif'):
                replace_name = files[:-9] + '2.ome.tif'
                os.rename(pathlib.Path(inpDir, files),
                          pathlib.Path(inpDir, replace_name))
    return inpDir




def nyxus_exe(inpDir:pathlib.Path,
              segDir:pathlib.Path,
              outDir:pathlib.Path,
              plate:str) -> pathlib.Path:
    
    """Run nyxus executable to extract features from images
    Args:
        inpDir (Path): Path to intensity images
        segDir (Path): Path to labelled images
        outDir (Path): Path to output folder
        
    Returns:
        paths to output folder containing CSV's
    """
    

    pluginName="nyxus"    
    outpath = create_output_folder(outDir, pluginName, plate)

    features = "*ALL_INTENSITY*,*basic_morphology*"
    filePattern=f"p{plate}.*\.*c2\.ome\.tif"
    csvFile="singlecsv"
    pixelsPerCentimeter=8361.2


    ARGS = {'verbosity':int(0),
            'intDir': f'{inpDir}',
            'segDir': f'{segDir}',
            'outDir': f'{outpath}',
            'filePattern': f'{filePattern}',
            'csvFile': f'{csvFile}',
            'pixelsPerCentimeter':f'{pixelsPerCentimeter}',
            'features': features
        }
    command = [
                '/opt/shared/notebooks/eastman/nyxus_exe/nyxus-20220819.staticblosc-linux-exe'             
            ]
    command.extend(f'--{i}={o}' for i, o in ARGS.items())
    command = ' '.join(command)
         
    os.system(command)

    
    return 

@my_timer
def run_nyxus_exe(inpDir:pathlib.Path,
              segDir:pathlib.Path,
              outDir:pathlib.Path,
              plate:str,
              dryrun:bool=True)-> pathlib.Path:
    
    """Parallel processing nyxus executable to extract features from images accross multiple plates
    Args:
        inpDir (Path): Path to intensity images
        segDir (Path): Path to labelled images
        outDir (Path): Path to output folder
        plate (str): unique regex as identifier for each plate
        
    Returns:
        paths to output folder containing CSV's
    """
      
    if not dryrun:

        starttime = time.time()
        num_workers = (multiprocessing.cpu_count())
        p = multiprocessing.Pool(processes=num_workers)

        outpath = p.map(partial(nyxus_exe, 
                        inpDir, 
                        segDir,
                        outDir,
                        ), plate)
        p.close()
        p.join()

        endtime = round((time.time() - starttime) / 60, 3) 
    return 
    
    
@my_timer
def renameCSVs(inpDir:pathlib.Path, 
               dryrun:bool=True)-> pathlib.Path:
    
    '''Renaming of Combined Nyxus CSVs with unique plate identifier and deleting subfolders
    Args:
        inpDir (Path): Path to directory containg subfolder for each plate containing merged nyxus CSV
        
    Returns:
        paths to output folder containing renamed CSV's with unique plate identifier only
    
    '''
    flist = [os.path.join(root, name) for root, dirs, files in os.walk(inpDir, topdown=False) for name in files]
    
    if not dryrun:      
        for fl in sorted(flist):
            root , fname = os.path.split(fl)[0], os.path.split(fl)[1]
            dirname = os.path.split(root)[1]   
            newname = os.path.join(root, f'{dirname}.csv')
            os.rename(fl, newname)   
            dest = pathlib.Path(root).parent 
            src = pathlib.Path(newname)
            shutil.move(src, dest)  
            shutil.rmtree(src.parent)
        
    return inpDir

@my_timer
def csv_merger(inpDir:pathlib.Path,
              outDir:pathlib.Path,
              dim:str,
              sameRows:Optional[str]='true',
              dryrun:bool=True
              ) -> pathlib.Path:
    
    '''Run polus-csv-merger-plugin to merge CSV files
    Args:
        inpDir (Path): Path to directory containg CSV's
        ouDir (Path): Path to output directory
        dim (str) : Specify merging CSV's either by rows or columns
        sameRows (str, optional): Only merge csvs if they contain the same number of rows
    
    Returns:
        paths to output folder containing single merged CSV    
    '''     
    pluginName='polus-csv-merger-plugin'
    outpath = create_output_folder(outDir, 
                         pluginName,
                         plate=None)
    stripExtension='true'
    args = {'inpDir': f'{inpDir}',
            'stripExtension':f'{stripExtension}',
            'outDir': f'{outpath}',
            'dim': f'{dim}',
            'sameRows': f'{sameRows}'
        }
        
    if not dryrun:
        run_command(pluginName, args) 
  
    return outpath