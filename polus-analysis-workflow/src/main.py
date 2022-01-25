import os
from pathlib import Path
import argparse, logging, os
import time
from workflow import *

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
parser.add_argument('--DATA_DIR', dest='DATA_DIR', type=str,
                        help='DATA_DIR containing images or stiching vectors', required=True)

parser.add_argument('--FILEPATTREN', dest='FILEPATTREN', type=str,
                        help='FILEPATTREN used to parse images', required=True)   

parser.add_argument('--OUT_DIR', dest='OUT_DIR', type=str,
                        help='OUT_DIR used to collect outputs', required=True)   
# # Parse the arguments
args = parser.parse_args()
DATA_DIR = Path(args.DATA_DIR) 
logger.info('DATA_DIR = {}'.format(DATA_DIR))
FILEPATTREN = str(args.FILEPATTREN) 
logger.info('FILEPATTREN = {}'.format(FILEPATTREN))
OUT_DIR = Path(args.OUT_DIR) 
logger.info('OUT_DIR = {}'.format(OUT_DIR))


def main(DATA_DIR:Path,
         FILEPATTREN:str,
         OUT_DIR:Path,
         ) -> None:

        starttime= time.time()
        logger.info("Step1: FlatField Correction plugin is running")
        Run_FlatField_Correction(DATA_DIR, FILEPATTREN,OUT_DIR, dryrun=True)
        #logger.info("Step1: Finished Running FlatField Estimation")
        #logger.info("Step2: Apply_FlatField_Correction plugin is running")
        # Apply_FlatField_Correction(DATA_DIR, FILEPATTREN,OUT_DIR, dryrun=False)
        #logger.info("Step2: Finished Running ApplyFlatField_Correction plugin")
        #logger.info("Step3: Montage plugin is running")
        #Run_Montage(DATA_DIR, OUT_DIR, dryrun:bool=False)
        #logger.info("Step3: Finished Running Montage plugin")
        #logger.info("Step4: Recycle_Vector plugin is running")
        #Run_Recycle_Vector(DATA_DIR,STICH_DIR,OUT_DIR, dryrun=False)
        #logger.info("Step4: Finished Running Recycle_Vector plugin")
        #logger.info("Step5: Image_Assembler plugin is Running ")
        #Run_Image_Assembler(DATA_DIR,STICH_DIR,OUT_DIR, dryrun=False)
        #logger.info("Step5: Finished Running Image_Assembler plugin")
        #logger.info("Step6: Precompute_Slide plugin is Running ")
        #Run_precompute_slide(DATA_DIR, IMAGETYPE='image', OUT_DIR, dryrun=False)
        #logger.info("Step6: Finished Running Precompute_Slide plugin")
        #logger.info("Step7: Run_SplineDist plugin is Running ")
        #Run_SplineDist(DATA_DIR, MODEL_DIR, OUT_DIR, dryrun:bool=False)
        #logger.info("Step7: Finished Running Run_SplineDist plugin")
        #logger.info("Step8: Nyxus plugin is Running ")
        #Run_Nyxus(DATA_DIR, SEG_DIR, FEATURES_TYPE='labels', OUT_DIR, dryrun:bool=False)
        #logger.info("Step8: Finished Running Nyxus plugin")
        #logger.info("Step9: Imagenet_Model_Featurization plugin is Running ")
        #Run_Imagenet_Model_Featurization(DATA_DIR,OUT_DIR,model,resolution,dryrun=False)
        #logger.info("Step9: Finished Running Imagenet_Model_Featurization plugin")
        #logger.info("Step10: DeepProfiler plugin is Running ")
        #Run_DeepProfiler(DATA_DIR,LABEL_DIR, FEAT_DIR,model,batchSize, OUT_DIR, dryrun=False)
        #logger.info("Step10: Finished Running DeepProfiler plugin")
            
        endtime = (time.time() - starttime)/60
        logger.info(f'Time taken to finished Step1-2: {endtime}')
  
            

if __name__=="__main__":
    main(DATA_DIR=DATA_DIR,
         FILEPATTREN=FILEPATTREN,
         OUT_DIR=OUT_DIR)

