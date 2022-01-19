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
# # Parse the arguments
args = parser.parse_args()
logger.info('DATA_DIR = {}'.format(DATA_DIR))
FILEPATTREN = str(args.FILEPATTREN) 
logger.info('FILEPATTREN = {}'.format(FILEPATTREN))


def main(DATA_DIR:Path,
         FILEPATTREN:str,
         ) -> None:


        starttime= time.time()
        logger.info("Step1: FlatField Correction plugin is running")
        Run_FlatField_Correction(DATA_DIR, FILEPATTREN)
        outpath = os.path.join(os.path.split(DATA_DIR)[0], 'basic-flatfield-correction-outputs')
        logger.info("Step2: Input Data Directory for Apply_FlatField_Correction".format(outpath))
        logger.info(f'Time taken to finished Step1-2: {outpath}')
        logger.info("Step2: Input Data Directory for Apply_FlatField_Correction".format(outpath))
        logger.info("Step2: Apply_FlatField_Correction plugin is running")
        Apply_FlatField_Correction(outpath,  FILEPATTREN)
            
        endtime = (time.time() - starttime)/60
        logger.info(f'Time taken to finished Step1-2: {endtime}')
        
                
            




if __name__=="__main__":
    main(DATA_DIR=DATA_DIR,
         FILEPATTREN=FILEPATTREN)

