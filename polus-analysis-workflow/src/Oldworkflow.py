  #### Old workflow      




def create_output_folder(DATA_DIR:Path,
                         outname:str) -> None:
    outpath = Path(os.path.split(DATA_DIR)[0], outname)
    if not os.path.exists(outpath):
        outdir = os.makedirs(Path(outpath))
        print(outdir)
        logger.info(f'{outname} directory is created')
    else:
        logger.info(f'{outname} directory already exists')


def assigning_pathnames(DATA_DIR:Path):
    pattern = re.compile("/")
    pattrenlist = [x for x in pattern.split(DATA_DIR) if x]
    DATA_DIRNAME = pattrenlist[-1]
    ROOT_DIR = '/'+'/'.join(pattrenlist[0:-1])
    TARGET_DIR = '/'+ pattrenlist[-2]
    return DATA_DIRNAME, ROOT_DIR, TARGET_DIR


        
def pull_docker_image(PLUGIN_NAME:str, 
                    VERSION:str, TAG:str)-> None:
    command = ['docker', 'pull', f'{TAG}/{PLUGIN_NAME}:{VERSION}'] 
    command = " ".join(command)
    os.system(command)
    logger.info(f'{PLUGIN_NAME} Pulling docker image')

def naming_outputfolder(PLUGIN_NAME:str):
    outname = PLUGIN_NAME.split('-')[1:-1]
    if not outname:
        outname = PLUGIN_NAME + '-' + 'outputs'
    else:
        outname = "-".join(outname) + '-' + 'outputs'
    return outname



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



def Run_FlatField_Correction(DATA_DIR:Path,
                              VERSION:Optional[str] = None)-> None:
    VERSION='1.2.8'  
    PLUGIN_NAME='polus-basic-flatfield-correction-plugin'
    TAG='labshare'
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)
    
    inpDir = Path(TARGET_DIR, 'images')
    filePattern = 'x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
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
    run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)

  

# DATA_DIR='/home/ec2-user/data/images'

#Run_FlatField_Correction(DATA_DIR)


def Apply_FlatField_Correction(DATA_DIR:Path,  VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-apply-flatfield-plugin'
    VERSION='1.0.6'
    TAG='labshare' 
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

    darkPattern='x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_darkfield.ome.tif'
    ffDir=Path(TARGET_DIR, DATA_DIRNAME, 'images')
    brightPattern='x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}_flatfield.ome.tif'
    imgDir=Path(TARGET_DIR, 'images')
    imgPattern='x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
    #photoPattern=''

    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'imgDir': imgDir,
            'imgPattern': imgPattern,
            'ffDir': ffDir,
            'brightPattern': brightPattern,
            'outDir': outDir,
            'darkPattern': darkPattern 
    }
    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)

    
# DATA_DIR = '/home/ec2-user/data/flatfield-correction-outputs'
# Apply_FlatField_Correction(DATA_DIR)


def Run_Montage(DATA_DIR:Path, VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-montage-plugin'
    VERSION='0.3.0'
    TAG='labshare' 
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

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
 
    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)

# DATA_DIR='/home/ec2-user/data/apply-flatfield-outputs'
# Run_Montage(DATA_DIR)


def Run_Recycle_Vector(DATA_DIR:Path, STICHDIR_NAME:str, VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-recycle-vector-plugin'
    VERSION='1.4.3'
    TAG='labshare' 
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

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
 
    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)
# DATA_DIR = '/home/ec2-user/data/corrected'
# STICHDIR_NAME = 'montage'
# Run_Recycle_Vector(DATA_DIR, STICHDIR_NAME)


def Run_Image_Assembler(DATA_DIR:Path, STICHDIR_NAME:str, VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-image-assembler-plugin'
    VERSION='1.1.2'
    TAG='labshare' 
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

    stitchPath=Path(TARGET_DIR, STICHDIR_NAME)
    imgPath=Path(TARGET_DIR, DATA_DIRNAME)
    timesliceNaming='false'
    # Output paths
    outDir = f'{TARGET_DIR}/{outname}'

    ARGS = {
            'stitchPath': stitchPath,
            'imgPath': imgPath,
            'outDir': outDir,
            'timesliceNaming': timesliceNaming,\
           
    }
 
    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)

def Run_precompute_slide(DATA_DIR:Path, IMAGETYPE:str, VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-precompute-slide-plugin'
    VERSION='1.3.12'
    TAG='labshare' 
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

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
 
    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)
# DATA_DIR = '/home/ec2-user/data/image-assembler-outputs'
# IMAGETYPE = 'segmentation'  
# Run_precompute_slide(DATA_DIR, IMAGETYPE)

def Run_SplineDist(DATA_DIR:Path, MODEL_DIRNAME:str, VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-splinedist-inference-plugin'
    VERSION='eastman03'
    TAG='labshare'    
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)

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

    run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)
# DATA_DIR = '/home/ec2-user/data/apply-outputs'
# Run_SplineDist(DATA_DIR, MODEL_DIRNAME='model')


def Run_Nyxus(DATA_DIR:Path, SEG_DIR:Path, FEATURES_TYPE:str, VERSION:Optional[str] = None):
    PLUGIN_NAME='nyxus'
    VERSION='0.2.4'
    TAG='polusai'    
    outname=naming_outputfolder(PLUGIN_NAME)
    outname = outname + '_' + FEATURES_TYPE
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    SEG_DIRNAME, _, _ = assigning_pathnames(SEG_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)
    
    intDir=Path(TARGET_DIR, DATA_DIRNAME)
    if FEATURES_TYPE == 'Images':
        segDir = intDir
    else:  
        segDir=Path(TARGET_DIR, SEG_DIRNAME)
    filePattern='.*c2\.ome\.tif'
    csvFile='singlecsv'
    #features='*ALL_INTENSITY*'
    features='*BASIC_MORPHOLOGY*'
    outDir = f'{TARGET_DIR}/{outname}'
    

    ARGS = {
            'intDir': intDir,
            'segDir': segDir,
            'outDir': outDir,
            'filePattern': filePattern,
            'csvFile': csvFile,
            'features': features
    }

    run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)
    

# DATA_DIR='/home/ec2-user/data/images'

# SEG_DIR='/home/ec2-user/data/labels'

# Run_Nyxus(DATA_DIR, SEG_DIR, FEATURES_TYPE = 'labels')

def Run_Imagenet_Model_Featurization(DATA_DIR:Path, 
                                    model:str, 
                                    resolution:str, 
                                    VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-imagenet-model-featurization-plugin'
    VERSION='0.1.2'
    TAG='labshare'    
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)
    
    inpDir=Path(TARGET_DIR, DATA_DIRNAME)
    outDir = f'{TARGET_DIR}/{outname}'
    
    ARGS = {
            'inpDir': inpDir,
            'outDir':outDir,
            'model': model,
            'resolution': resolution
    }

    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)


DATA_DIR='/home/ec2-user/data/images'
#Run_Imagenet_Model_Featurization(DATA_DIR, model='VGG19', resolution='500x500')



def Run_DeepProfiler(DATA_DIR:Path,
                    LABEL_DIR:Path,
                    FEAT_DIR:Path,
                    model:str, 
                    batchSize:int, 
                    VERSION:Optional[str] = None):
    PLUGIN_NAME='polus-deep-profiler-plugin'
    VERSION='0.1.6'
    TAG='polusai'    
    outname=naming_outputfolder(PLUGIN_NAME)
    DATA_DIRNAME, ROOT_DIR, TARGET_DIR = assigning_pathnames(DATA_DIR)
    LABEL_DIRNAME, _, _ = assigning_pathnames(LABEL_DIR)
    FEAT_DIRNAME, _, _ = assigning_pathnames(FEAT_DIR)
    create_output_folder(DATA_DIR, outname)
    pull_docker_image(PLUGIN_NAME, VERSION, TAG)
    
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

    return run_command(PLUGIN_NAME, ROOT_DIR, TARGET_DIR, TAG, VERSION, ARGS)
DATA_DIR='/home/ec2-user/data/images'
LABEL_DIR='/home/ec2-user/data/labels'
FEAT_DIR='/home/ec2-user/data/nyxus-outputs_labels'
model='VGG16'
batchSize=8

# Run_DeepProfiler(DATA_DIR,
#                     LABEL_DIR,
#                     FEAT_DIR,
#                     model,
#                     batchSize)