# Base packages
import argparse, logging, re, typing, pathlib, queue

# 3rd party packages
import filepattern, numpy

# Class/function imports
from bfio import BioReader,BioWriter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from multiprocessing import Queue, cpu_count

# Global variable to scale number of processing threads dynamically
max_threads = max([cpu_count()//2,1])
available_threads = Queue(max_threads)

# Set logger delay times
process_delay = 30    # process status update period in seconds
thread_delay = 10     # thread status update period in seconds

for _ in range(max_threads):
    available_threads.put(2)

# length/width of the chunk each _merge_layers thread processes at once
# Number of useful threads is limited
chunk_size = 8192
useful_threads = (chunk_size // BioReader._TILE_SIZE) ** 2

def initialize_queue(processes: Queue,
                     file_pattern: filepattern.FilePattern) -> None:
    """Initialize global variables for each process

    This function is called when each worker process is started in the
    ProcessPoolExecutor. Within each worker process, two global variables are
    defined: ``available_threads`` (multiprocessing.Queue) and ``fp``
    (filepattern.FilePattern). The ``available_threads`` variable defines
    globally available threads across processes. The ``fp`` object contains the
    parsed files in the input directory, which is useful to provide globally so
    that the input file directory does not need to parsed in every process. This
    is especially beneficial in instances where there are large numbers of files
    in the input directory.

    Args:
        processes: The globally available threads.
        file_pattern: The parsed input file directory.
    """
    global available_threads
    global fp
    available_threads = processes
    fp = file_pattern

def buffer_image(image_path: pathlib.Path,
                 supertile_buffer: numpy.ndarray,
                 Xi: typing.Tuple[int,int],
                 Yi: typing.Tuple[int,int],
                 Xt: typing.Tuple[int,int],
                 Yt: typing.Tuple[int,int],
                 local_threads: queue.Queue) -> None:
    """buffer_image Load and image and store in buffer

    This method loads an image and stores it in the appropriate position in the
    buffer based on the stitching vector coordinates. It is intended to be used
    as a thread.

    Args:
        image_path: Path of image to load
        supertile_buffer: A supertile storing multiple images
        Xi: Xmin and Xmax of pixels to load from the image
        Yi: Ymin and Ymax of pixels to load from the image
        Xt: X position within the buffer to store the image
        Yt: Y position within the buffer to store the image
        local_threads: Used to determine if threads are available
    """

    # Get available threads
    active_threads = local_threads.get()

    # Load the image
    with BioReader(image_path,max_workers=active_threads) as br:
        image = br[Yi[0]:Yi[1],Xi[0]:Xi[1],0:1,0,0] # only get the first z,c,t layer

    # Free threads
    local_threads.put(active_threads)

    # Put the image in the buffer
    supertile_buffer[Yt[0]:Yt[1],Xt[0]:Xt[1],...] = image

def make_tile(x_min: int,
              x_max: int,
              y_min: int,
              y_max: int,
              parsed_vector: dict,
              local_threads: queue.Queue,
              bw: BioWriter) -> None:
    """Create a supertile from images and save to file

    This method builds a supertile, which is a section of the image defined by
    the global variable ``chunk_size`` and is composed of multiple smaller tiles
    defined by the ``BioReader._TILE_SIZE``. Images are stored on disk as
    compressed chunks that are ``_TILE_SIZE`` length and width, and the upper
    left pixel of a tile is always a multiple of ``_TILE_SIZE``. To prevent
    excessive file loading and to ensure files are properly placed, supertiles
    are created from smaller images and saved all at once.

    Args:
        x_min: Minimum x bound of the tile
        x_max: Maximum x bound of the tile
        y_min: Minimum y bound of the tile
        y_max: Maximum y bound of the tile
        parsed_vector: The result of _parse_vector
        local_threads: Used to determine the number of concurrent threads to run
        bw: The output file object

    """

    # Get the data type
    with BioReader(parsed_vector['filePos'][0]['file']) as br:
        dtype = br.dtype

    # initialize the supertile
    template = numpy.zeros((y_max-y_min,x_max-x_min,1,1,1),dtype=dtype)

    # get images in bounds of current super tile
    with ThreadPoolExecutor(useful_threads//2) as executor:
        for f in parsed_vector['filePos']:

            # check that image is within the x-tile bounds
            if (f['posX'] >= x_min and f['posX'] <= x_max) \
                or (f['posX']+f['width'] >= x_min and f['posX']+f['width'] <= x_max):

                # check that image is within the y-tile bounds
                if (f['posY'] >= y_min and f['posY'] <= y_max) \
                    or (f['posY']+f['height'] >= y_min and f['posY']+f['height'] <= y_max):

                    # get bounds of image within the tile
                    Xt = [max(0,f['posX']-x_min)]
                    Xt.append(min(x_max-x_min,f['posX']+f['width']-x_min))
                    Yt = [max(0,f['posY']-y_min)]
                    Yt.append(min(y_max-y_min,f['posY']+f['height']-y_min))

                    # get bounds of image within the image
                    Xi = [max(0,x_min - f['posX'])]
                    Xi.append(min(f['width'],x_max - f['posX']))
                    Yi = [max(0,y_min - f['posY'])]
                    Yi.append(min(f['height'],y_max - f['posY']))

                    # Start a thread to load and store the image
                    executor.submit(buffer_image,f['file'],template,Xi,Yi,Xt,Yt,local_threads)

    # Save the image
    bw[y_min:y_max,x_min:x_max,:1,0,0] = template

def get_number(s: typing.Any) -> typing.Union[int,typing.Any]:
    """ Check that s is number

    This function checks to make sure an input value is able to be converted
    into an integer. If it cannot be converted to an integer, the original
    value is returned.

    Args:
        s: An input string or number
    Returns:
        Either ``int(s)`` or return the value if s cannot be cast
    """
    try:
        return int(s)
    except ValueError:
        return s

def _parse_stitch(stitchPath: pathlib.Path,
                  timepointName: bool = False) -> dict:
    """ Load and parse image stitching vectors

    This function parses the data from a stitching vector, then extracts the
    relevant image sizes for each image in the stitching vector to obtain a
    stitched image size. This function also infers an output file name.

    Args:
        stitchPath: A path to stitching vectors
        timepointName: Use the vector timeslice as the image name
    Returns:
        Dictionary with keys (width, height, name, filePos)
    """
    
    logger = logging.getLogger('asmbl')

    # Initialize the output
    out_dict = { 'width': int(0),
                 'height': int(0),
                 'name': '',
                 'filePos': []}

    # Try to parse the stitching vector using the infered file pattern
    if fp.pattern != '.*':
        vp = filepattern.VectorPattern(stitchPath,fp.pattern)
        unique_vals = {k.upper():v for k,v in vp.uniques.items() if len(v)==1}
        files = fp.get_matching(**unique_vals)

    else:
        
        # Try to infer a pattern from the stitching vector        
        try:
            vector_files = filepattern.VectorPattern(stitchPath,'.*')
            pattern = filepattern.infer_pattern([v[0]['file'] for v in vector_files()])
            vp = filepattern.VectorPattern(stitchPath,pattern)
            
        # Fall back to universal filepattern
        except ValueError:
            vp = filepattern.VectorPattern(stitchPath,'.*')
            
        files = fp.files

    file_names = [f['file'].name for f in files]

    for file in vp():

        if file[0]['file'] not in file_names:
            continue

        stitch_groups = {k:get_number(v) for k,v in file[0].items()}
        stitch_groups['file'] = files[0]['file'].with_name(stitch_groups['file'])

        # Get the image size
        stitch_groups['width'], stitch_groups['height'] = BioReader.image_size(stitch_groups['file'])

        # Set the stitching vector values in the file dictionary
        out_dict['filePos'].append(stitch_groups)

    # Calculate the output image dimensions
    out_dict['width'] = max([f['width'] + f['posX'] for f in out_dict['filePos']])
    out_dict['height'] = max([f['height'] + f['posY'] for f in out_dict['filePos']])

    # Generate the output file name
    if timepointName:
        global_regex = ".*global-positions-([0-9]+).txt"
        name = re.match(global_regex,pathlib.Path(stitchPath).name).groups()[0]
        name += '.ome.tif'
        out_dict['name'] = name
    else:
        # Try to infer a good filename
        try:
            out_dict['name'] = vp.output_name()
            logger.info(f'{out_dict["name"]}: Inferred output file name from vector.')

        # A file name couldn't be inferred, default to the first image name
        except:
            logger.info(f'{out_dict["name"]}: Could not infer output file name from vector, using first file name in the stitching vector as an output file name.')
            for file in vp():
                out_dict['name'] = file[0]['file']
                break

    return out_dict

def assemble_image(vector_path: pathlib.Path,
                   out_path: pathlib.Path) -> None:
    """Assemble a 2-dimensional image
    
    This method assembles one image from one stitching vector. It is intended
    to run as a process to parallelize stitching of multiple images.
    
    The basic approach to stitching is:
    1. Parse the stitching vector and abstract the image dimensions
    2. Generate a thread for each subsection (supertile) of an image.

    Args:
        vector_path: Path to the stitching vector
        out_path: Path to the output directory
    """

    logger = logging.getLogger('asmbl')
    logger.setLevel(logging.INFO)

    # Get globally available threads, defined in initialize_queue
    active_threads = available_threads.get()

    # Set up a local thread queue
    local_threads = queue.Queue()
    for _ in range(active_threads//2):
        local_threads.put(2)

    # Parse the stitching vector
    parsed_vector = _parse_stitch(vector_path,timesliceNaming)

    # Initialize the output image
    with BioReader(parsed_vector['filePos'][0]['file']) as br:
        bw = BioWriter(out_path.joinpath(parsed_vector['name']),
                       metadata=br.metadata,
                       max_workers=active_threads)
        bw.x = parsed_vector['width']
        bw.y = parsed_vector['height']

    # Assemble the images
    logger.info(f'{parsed_vector["name"]}: Begin assembly with {active_threads} threads')
    threads = []
    with ThreadPoolExecutor(1) as executor:
        for x in range(0, parsed_vector['width'], chunk_size):
            X_range = min(x+chunk_size,parsed_vector['width']) # max x-pixel index in the assembled image
            for y in range(0, parsed_vector['height'], chunk_size):
                Y_range = min(y+chunk_size,parsed_vector['height']) # max y-pixel index in the assembled image

                threads.append(executor.submit(make_tile,x,X_range,y,Y_range,parsed_vector,local_threads,bw))

        done, not_done = wait(threads,timeout=0)

        while len(not_done) > 0:

            # See if more threads are available
            new_threads = 0
            try:
                while active_threads + new_threads < useful_threads:
                    new_threads += available_threads.get(block=False)
            except:
                pass

            if new_threads > 0:
                logger.info(f'{parsed_vector["name"]}: Increasing threads from {active_threads} to {active_threads+new_threads}')
                active_threads += new_threads
                bw.max_workers = active_threads
                for _ in range(new_threads//2):
                    local_threads.put(2)

            done, not_done = wait(threads,timeout=thread_delay)

            logger.info('{}: Progress: {:7.3f}%'.format(parsed_vector['name'],100*len(done)/len(threads)))

    # Free the threads for other processes
    for _ in range(active_threads//2):
        available_threads.put(2)

    logger.info(f'{parsed_vector["name"]}: Closing image...')
    bw.close()

if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    '''Parse arguments'''
    # Setup the argument parsing
    parser = argparse.ArgumentParser(prog='main', description='Assemble images from a single stitching vector.')
    parser.add_argument('--stitchPath', dest='stitchPath', type=str,
                        help='Complete path to a stitching vector', required=True)
    parser.add_argument('--imgPath', dest='imgPath', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    parser.add_argument('--timesliceNaming', dest='timesliceNaming', type=str,
                        help='Use timeslice number as image name', required=False)

    # Parse the arguments
    args = parser.parse_args()
    imgPath = pathlib.Path(args.imgPath).resolve()
    if imgPath.joinpath('images').is_dir():
        imgPath = imgPath.joinpath('images')
    logger.info('imgPath: {}'.format(imgPath))
    outDir = pathlib.Path(args.outDir).resolve()
    logger.info('outDir: {}'.format(outDir))
    timesliceNaming = args.timesliceNaming == 'true'
    logger.info('timesliceNaming: {}'.format(timesliceNaming))
    stitchPath = args.stitchPath
    logger.info('stitchPath: {}'.format(stitchPath))

    '''Setup stitching variables/objects'''
    # Get a list of stitching vectors
    vectors = [p for p in pathlib.Path(stitchPath).iterdir()]
    vectors.sort()

    # Try to infer a filepattern from the files on disk for faster matching later
    try:
        pattern = filepattern.infer_pattern([f.name for f in imgPath.iterdir()])
        logger.info(f'Inferred file pattern: {pattern}')
        fp = filepattern.FilePattern(imgPath,pattern)

    # Pattern inference didn't work, so just get a list of files
    except:
        logger.info(f'Unable to infer pattern, defaulting to: .*')
        fp = filepattern.FilePattern(imgPath,'.*')

    '''Run stitching jobs in separate processes'''
    processes = []
    with ProcessPoolExecutor(max_threads,initializer=initialize_queue,initargs=(available_threads,fp)) as executor:

        for v in vectors:
            # Check to see if the file is a valid stitching vector
            if 'img-global-positions' not in v.name:
                continue
            
            processes.append(executor.submit(assemble_image,v,outDir))

        # Wait for processes to finish, providing periodic updates
        not_done = []
        while len(not_done) > 0:

            done, not_done = wait(processes,timeout=process_delay)

            logger.info('Total Progress: {:6.2f}%'.format(100*len(done)/len(processes)))
