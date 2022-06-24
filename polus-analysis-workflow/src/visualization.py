import os, re, time, argparse, logging
from pathlib import Path
from workflow import *
from analysis import *
from typing import Optional
from polus.data import collections



#Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename='log.log', filemode='w')
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
parser.add_argument('--platesNum', dest='platesNum', type=int,
                        help='Total number of 384-well plates', required=True)         
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
platesNum= int(args.platesNum) 
logger.info('platesNum = {}'.format(platesNum))
outDir = args.outDir
logger.info('outDir = {}'.format(outDir))







def main(data:str,
        filePattern:str,
        groupBy:str,
        modelDir:Path,
        platesNum:int,
        outDir:Path,
        model:Optional[str]=None,
        resolution:Optional[str]=None,
         ) -> None:
        logging.basicConfig(filename='file.log', level=logging.INFO)
    
        starttime= time.time()
      
        if platesNum > 1:
            plates = [str(i).zfill(2) for i in range(0, platesNum+1)]
        else:
            plates = [''.join([s for s in filePattern][1:3])]
            
        for plate in plates:
        
            filePattern = '{0}{1}_{2}'.format('p', plate, '_'.join((filePattern.split('_')[1:])))         
            logger.info("Running Workflow for plate:{plate}, filePattern:{filePattern}")
            # logger.info("Step1: Loading image data collection")
            # inpDir = collections[data].standard.intensity.path
            # assert inpDir.exists(), f'Directory does not exist: {inpDir}'
            # fltime= time.time()
            # logger.info("Step2: FlatField Correction plugin is running")
            # outpath  = Run_FlatField_Correction(inpDir, filePattern,groupBy, outDir, dryrun=False)
            # flattime = (time.time() - fltime)/60
            # logger.info(f"FlatField Correction: {flattime}")
            # logger.info("Step2: FlatField Correction plugin is running")
            # assert outpath.exists(), f'Directory does not exist: {outpath}'
            # logger.info("Step2: Finished Running FlatField Estimation")
            # aptime= time.time()
            # logger.info("Step3: Apply_FlatField_Correction plugin is running")
            # corrDir = ApplyFlatfield(inpDir=inpDir, filePattern=filePattern,outDir=outDir,ffDir=outpath, dryrun=False)  
            # applytime = (time.time() - aptime)/60
            # logger.info(f"Apply_FlatField_Correction: {applytime}")
           
            # assert corrDir.exists(), f'Directory does not exist: {corrDir}'
            # logger.info("Step3: Finished Running ApplyFlatField_Correction plugin") 
            # montime= time.time()
            # logger.info("Step4: Montage plugin is running")
            # outpath = Run_Montage(inpDir=corrDir, filePattern=filePattern, outDir=outDir,dryrun=False)
            # logger.info("Step4: Finished Running Montage plugin")
            # logger.info(f"Finished Running Montage: {(time.time() - montime)/60}")
            # rectime= time.time()
            # logger.info("Step5: Recycle_Vector plugin is running")
            # outpath = Recycle_Vector(inpDir=corrDir, stitchDir=outpath, filePattern=filePattern, outDir=outDir, dryrun=False)
            # logger.info("Step5: Finished Running Recycle_Vector plugin")
            # logger.info(f"Finished Running Montage: {(time.time() - rectime)/60}")
            # assemtime= time.time()
            # logger.info("Step6: Image_Assembler plugin is Running ")
            # outpath = Image_Assembler(inpDir=corrDir, stitchPath=outpath, outDir=outDir, dryrun=False)
            # logger.info("Step5: Finished Running Image_Assembler plugin")
            # logger.info(f"Finished Running Image_Assembler: {(time.time() - assemtime)/60}")
            logger.info("Step6: Precompute_Slide plugin is Running ")
            outpath = '/home/ec2-user/data/Image_Assembler_outputs'
            precomtime= time.time()
            outpath = precompute_slide(inpDir=outpath, plate=plate, pyramidType='DeepZoom', imageType='image', outDir=outDir, dryrun=False)
            # logger.info("Step6: Finished Running Precompute_Slide plugin")
            # logger.info(f"Finished Running Precompute_Slide plugin: {(time.time() - precomtime)/60}")
            endtime = (time.time() - starttime)/60
            logger.info(f'Total time of visualization pipeline is {endtime} minutes')
    
if __name__=="__main__":
    main(data=data,
        filePattern=filePattern,
        groupBy=groupBy,
        modelDir=modelDir,
        platesNum=platesNum,
        outDir=outDir,
        model=model,
        resolution=resolution)

