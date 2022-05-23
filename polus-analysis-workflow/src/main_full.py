import os
from pathlib import Path
import argparse, logging, os
from pickle import FALSE
import time
from workflow import *
from typing import Optional
from polus.data import collections




#Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

# ''' Argument parsing '''
logger.info("Parsing arguments...")
parser = argparse.ArgumentParser(prog='main', description='Analysis Workflow')    
#     # Input arguments
parser.add_argument('--data', dest='data', type=str,
                        help='Data to be used for processing', required=True)

parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='filePattern used to parse images', required=True) 
parser.add_argument('--groupBy', dest='groupBy', type=str,
                        help='groupBy for calculating flatfield correction', required=True) 
parser.add_argument('--model', dest='model', type=str,
                        help='choose the model for imagenet model featurization plugin', required=False) 
parser.add_argument('--resolution', dest='groupBy', type=str,
                        help='image resolution use for imagenet model featureization plugin', required=False) 
parser.add_argument('--modelDir', dest='modelDir', type=str,
                        help='Path to Tranined model', required=True)     
parser.add_argument('--outDir', dest='outDir', type=str,
                        help='outDir used to collect outputs', required=True)   
# # Parse the arguments
args = parser.parse_args()
data = str(args.data) 
logger.info('data = {}'.format(data))
filePattern = str(args.filePattern) 
logger.info('filePattern = {}'.format(filePattern))
groupBy = str(args.groupBy) 
logger.info('groupBy = {}'.format(groupBy))
model = str(args.model) 
logger.info('model = {}'.format(model))
resolution= str(args.model) 
logger.info('resolution = {}'.format(model))
modelDir = Path(args.modelDir) 
logger.info('modelDir = {}'.format(modelDir))
outDir = args.outDir
logger.info('outDir = {}'.format(outDir))


def main(data:str,
        filePattern:str,
        groupBy:str,
        modelDir:Path,
        outDir:Path,
        model:Optional[str]=None,
        resolution:Optional[str]=None
         ) -> None:
        starttime= time.time()

        logger.info("Step1: Loading image data collection")
        inpDir = collections[data].standard.intensity.path
        logger.info("Step2: FlatField Correction plugin is running")
        outpath  = Run_FlatField_Correction(inpDir, filePattern,groupBy, outDir, dryrun=True)
        logger.info("Step2: Finished Running FlatField Estimation")
        logger.info("Step3: Apply_FlatField_Correction plugin is running")
        corrDir = ApplyFlatfield(inpDir=inpDir, filePattern=filePattern,outDir=outDir,ffDir=outpath, dryrun=True)
        logger.info("Step3: Finished Running ApplyFlatField_Correction plugin")
        # logger.info("Step3: Montage plugin is running")
        # outpath = Run_Montage(corrDir, filePattern, outDir, dryrun=True)
        # logger.info("Step3: Finished Running Montage plugin")
        # logger.info("Step4: Recycle_Vector plugin is running")
        # outpath = Recycle_Vector(inpDir=corrDir, stitchDir=outpath, groupBy=groupBy, filePattern=filePattern, outDir=outDir, dryrun=True)
        # logger.info("Step4: Finished Running Recycle_Vector plugin")
        # logger.info("Step5: Image_Assembler plugin is Running ")
        # outpath = Image_Assembler(inpDir=corrDir, stitchPath=outpath, outDir=outDir, dryrun=True)
        # logger.info("Step5: Finished Running Image_Assembler plugin")
        # logger.info("Step6: Precompute_Slide plugin is Running ")
        # outpath = precompute_slide(inpDir=corrDir, filePattern=filePattern, imageType='image', outDir=outDir, dryrun=True)
        # logger.info("Step6: Finished Running Precompute_Slide plugin")
        # logger.info("Step7: Run_SplineDist plugin is Running ")
        # outpath = SplineDist(inpDir=corrDir, filePattern=filePattern, modelDir=modelDir, outDir=outDir, dryrun=True)
        # logger.info("Step7: Finished Running Run_SplineDist plugin")
        # logger.info("Step8: Imagenet_Model_Featurization plugin is Running ")
        # outpath = ImagenetModelFeaturization(inpDir=corrDir, model=model, resolution=resolution, outDir=outDir,dryrun=True)
        # logger.info("Step8: Finished Running Imagenet_Model_Featurization plugin")
        # segDir = cellposeInference(inpDir=corrDir, filePattern=filePattern, outDir=outDir, dryrun=True)
        
        logger.info("Step9: SMP_training_inference plugin is running ")
        filePattern='p01_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif'
        segDir = SMP_training_inference(inpDir=corrDir, filePattern=filePattern, modelDir=modelDir, outDir=outDir,dryrun=True)
        logger.info("Step9: Finished Running SMP_training_inference plugin")
        logger.info("Step10: FtlLabel plugin is running")
        segDir = '/home/ec2-user/data/polus_smp_training_outputs'
        segDir = FtlLabel(inpDir=segDir, outDir=outDir, dryrun=False)
        logger.info("Step10: Finished Running FtlLabel plugin")
        logger.info("Step11: Rename of files for channel")
        segDir = rename_files(inpDir=segDir)
        logger.info("Step11: Finish Renaming of files for channel")
        logger.info("Step12: Nyxus plugin is Running")
        filePattern='.*c2\.ome\.tif'
        outpath = Nyxus_exe(inpDir=corrDir, segDir=segDir, filePattern=filePattern, outDir=outDir, dryrun=False)
        logger.info("Step12: Finished Running Nyxus plugin")
        endtime = (time.time() - starttime)/60
        logger.info(f'Time taken to finished Step1-2: {endtime}')
  
            

if __name__=="__main__":
    main(data=data,
        filePattern=filePattern,
        groupBy=groupBy,
        modelDir=modelDir,
        outDir=outDir,
        model=model,
        resolution=resolution)

