


import os
import pandas as pd
import tifffile
import numpy as np
import cv2
import math
import ashlar.scripts.ashlar as ashlar
import re

def zen_OME_tiff(exported_directory, output_directory, channel_split = 3, cycle_split = 2):
    '''
    using this function is predicated on the fact that you are using the nilsson SOP for naming files. this only works if we have rather small sections. 
    '''

    import tifffile
    import os
    from os import listdir
    import pandas as pd
    import numpy as np
    from xml.dom import minidom

    # make directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # find files
    onlyfiles = listdir(exported_directory)
    onlytifs =  [k for k in onlyfiles if '.tif' in k]
    onlyfiles_df = pd.DataFrame(onlytifs)

    onlyfiles_split_tiles = onlyfiles_df[0].str.split('m',expand=True)
    onlyfiles_split_channel = onlyfiles_split_tiles[0].str.split('_',expand=True)

    tiles = list(np.unique(onlyfiles_split_tiles[1]))
    channels = list(np.unique(onlyfiles_split_channel[channel_split]))
    rounds = list(np.unique(onlyfiles_split_channel[cycle_split]))

    for i, round_number in enumerate(rounds):
        onlytifs_round_filt = [l for l in onlytifs if '_'+round_number+'_' in l]
        imgs = []
        for i in (tiles):
            stacked = np.empty((5, 2048, 2048))
            tile_filtered = [k for k in onlytifs_round_filt if i in k]
            for n,image_file in enumerate(tile_filtered):
                image_int = tifffile.imread(exported_directory +'/'+image_file)
                stacked[n] = image_int
            imgs.append(stacked)

        metadatafiles =  [k for k in onlyfiles if 'info.xml' in k]
        metadatafiles_filt =  [k for k in metadatafiles if '_'+round_number+'_' in k]

        for p, meta in enumerate(metadatafiles_filt):
            mydoc = minidom.parse(exported_directory +'/'+ meta)
            tile =[]
            x =[]
            y =[]
            items = mydoc.getElementsByTagName('Bounds')
            for elem in items:
                tile.append(int(elem.attributes['StartM'].value))
                x.append(float(elem.attributes['StartX'].value))
                y.append(float(elem.attributes['StartY'].value))
            unique_tiles = list(np.unique(tile))
            x_reformatted = (x[:len(unique_tiles)])    
            y_reformatted = (y[:len(unique_tiles)])     
            dictionary = {'x': x_reformatted, 'y': y_reformatted}  

            df = pd.DataFrame(dictionary) 
            positions = np.array(df).astype(int)

            pixel_size = 0.1625
            with tifffile.TiffWriter(output_directory+'/cycle_'+str(round_number)+'.ome.tif', bigtiff=True) as tif:
                for img, p in zip(imgs, positions.astype(int)):
                    metadata = {
                        'Pixels': {
                            'PhysicalSizeX': pixel_size,
                            'PhysicalSizeXUnit': 'µm',
                            'PhysicalSizeY': pixel_size,
                            'PhysicalSizeYUnit': 'µm'
                        },
                        'Plane': {
                            'PositionX': [p[0]*pixel_size]*img.shape[0],
                            'PositionY': [p[1]*pixel_size]*img.shape[0]
                        }

                    }
                    tif.write(img.astype('uint16'), metadata=metadata)

def leica_mipping(input_dirs, output_dir_prefix, image_dimension = [2048, 2048]):

    '''

    the input is a list of the file paths to the files.
    used to MIP files from leica when exported as tiffs. 




    '''
    from os import listdir
    from os.path import isfile, join
    import tifffile
    from xml.dom import minidom
    import pandas as pd
    import numpy as np
    import os
    from tifffile import imread
    from tqdm import tqdm
    import re
    import shutil
    # only needed on linux

    input_dirs_reformatted = []
    for i in input_dirs: 
        i = i.replace("%20", " ") # needs to be done in linux thanks to the spaces
        input_dirs_reformatted.append(i)

    for ö,i in enumerate(input_dirs_reformatted):
        files = os.listdir(i)
        tifs =  [k for k in files if 'dw' not in k] # filter for deconvolved images
        tifs =  [k for k in tifs if '.tif' in k]
        tifs =  [k for k in tifs if '.txt' not in k]
        tifs =  [k for k in tifs if 'Corrected' in k]
        split_underscore = pd.DataFrame(tifs)[0].str.split('--', expand = True)
        regions_int = list(split_underscore[0].unique())
        regions = []
        for j in regions_int:
            regions.append(j)
        regions = list(np.unique(regions))

        # IF THE SCAN IS BIG ENOUGH, THE SECTION WILL BE DIVIDED INTO DIFFERENT REGIONS. THEREFORE WE NEED TO CHECK THIS IN THE FILES
        for region in regions: 
            tifs_filt =  [k for k in tifs if region in k]
            bases = str((ö)+1) #[i.split('/')[5].split('cycle')[1]]
            split_underscore = pd.DataFrame(tifs_filt)[0].str.split('--', expand = True)
            # GET TILES
            tiles = sorted(split_underscore[1].unique())
            tiles_df = pd.DataFrame(tiles)
            tiles_df['indexNumber'] = [int(i.split('e')[-1]) for i in tiles_df[0]]
            tiles_df.sort_values(by = ['indexNumber'], ascending = [True], inplace = True)
            tiles_df.drop('indexNumber', 1, inplace = True)
            tiles = list(tiles_df[0])

            # GET CHANNELS
            channels = split_underscore[3].unique()
            if len(regions) == 1:
                output_dir = output_dir_prefix
                folder_output = output_dir + '/preprocessing/mipped/'
            else: 
                output_dir = output_dir_prefix + '_R'+region.split('Region')[1].split('_')[0]
                folder_output = output_dir + '/preprocessing/mipped/'
            if not os.path.exists(folder_output):
                os.makedirs(folder_output)

            for ååå, w in enumerate(sorted(bases)):
                imgs = []
                if not os.path.exists(folder_output +'/Base_'+w):
                            os.makedirs(folder_output +'/Base_'+w)
                try: 
                    file_to_copy = join(i,'Metadata',([k for k in os.listdir(join(i,'Metadata')) if region in k][0]))
                    #shutil.copytree(join(i,'Metadata'), join(folder_output,('Base_'+w),'MetaData'))
                    if not os.path.exists(join(folder_output,('Base_'+w),'MetaData')):
                            os.makedirs(join(folder_output,('Base_'+w),'MetaData'))
                    shutil.copy(file_to_copy, join(folder_output,('Base_'+w),'MetaData'))
                except FileExistsError:
                    print(' ')

                # LOOP OVER THE TILES TO MIP
                for _tile in tqdm(range(len(tiles))):
                    tile = tiles[_tile]
                    tile_for_name = re.split('(\d+)', tile)[1]
                    strings_with_substring = [string for string in os.listdir(folder_output +'/Base_'+w) if str(tile_for_name) in string]

                    # ENSURE THAT WE DO NOT CREATE FILES THAT HAVE ALREADY BEEN CREATED
                    #if len(strings_with_substring) < 4:
                    tifs_base_tile = [k for k in tifs_filt if str(tile)+'--' in k]
                    for å,z in enumerate(sorted(list(channels))):
                        tifs_base_tile_channel = [k for k in tifs_base_tile if str(z) in k]
                        # DEFINE IMAGE TO USE AS GROUND ZERO 
                        maxi = np.zeros((image_dimension[0],image_dimension[1]))
                        for n,q in enumerate(tifs_base_tile_channel):
                            
                            try:
                                im_array = imread(i + '/' +q)
                            except:
                                print('image corrupted, reading black file instead.')
                                im_array = np.zeros((image_dimension[0],image_dimension[1]))

                            inds = im_array > maxi # find where image intensity > max intensity
                            maxi[inds] = im_array[inds]
                        maxi = maxi.astype('uint16')
                        # WRITE FILE
                        tifffile.imwrite(folder_output +'/Base_'+w+'/Base_'+w+'_s'+str(tile_for_name)+'_'+z, maxi)
                    #else: 
                    #    continue

def leica_OME_tiff(directory_base, output_directory):

    import tifffile
    import numpy as np
    import os
    from os.path import join
    import tifffile
    import os
    from os import listdir
    import pandas as pd
    import numpy as np
    from xml.dom import minidom
    from pathlib import Path
    from tqdm import tqdm

    folders = os.listdir(directory_base)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for folder in folders:
        exported_directory = join(directory_base,folder)
        onlyfiles = listdir(exported_directory)
        onlytifs =  [k for k in onlyfiles if '.tif' in k]
        onlyfiles_df = pd.DataFrame(onlytifs)
        onlyfiles_split_tiles = onlyfiles_df[0].str.split('_s',expand=True)
        onlyfiles_split_channel = onlyfiles_split_tiles[1].str.split('_',expand=True)

        tiles = list(np.unique(onlyfiles_split_tiles[1].str.split('_',expand=True)[0]))
        tiles_df=pd.DataFrame(tiles)
        tiles_df['indexNumber'] = [int(i.split('e')[-1]) for i in tiles_df[0]]
        # Perform sort of the rows
        tiles_df.sort_values(by = ['indexNumber'], ascending = [True], inplace = True)
        # Deletion of the added column
        tiles_df.drop('indexNumber', 1, inplace = True)
        tiles = list(tiles_df[0])
        channels = list(np.unique(onlyfiles_split_channel[1]))
        rounds = list(np.unique(onlyfiles_split_tiles[0]))
        
        
        metadatafiles = listdir(join(exported_directory, 'MetaData'))
        metadatafiles =  [k for k in metadatafiles if 'IOManagerConfiguation.xlif' not in k]

        for p, meta in enumerate(metadatafiles):
            mydoc = minidom.parse(join(exported_directory, 'MetaData',meta) )
            tile =[]
            x =[]
            y =[]
            items = mydoc.getElementsByTagName('Tile')
            for elem in items:
                tile.append(int(elem.attributes['FieldX'].value))
                x.append(float(elem.attributes['PosX'].value))
                y.append(float(elem.attributes['PosY'].value))
            unique_tiles = list(np.unique(tile))
            x_reformatted = (x[:len(unique_tiles)])    
            y_reformatted = (y[:len(unique_tiles)])     
            dictionary = {'x': x_reformatted, 'y': y_reformatted}  

            df = pd.DataFrame(dictionary)
            df['x'] =((df.x-np.min(df.x))/.000000321) + 1
            df['y'] =((df.y-np.min(df.y))/.000000321) + 1
            positions = np.array(df).astype(int)
            
        with tifffile.TiffWriter((output_directory +'/'+ folder + '.ome.tiff'), bigtiff=True) as tif:
            for i in tqdm(range(len(tiles))):
                position = positions[i]
                tile = tiles[i]

                tile_filtered = [k for k in onlytifs if 's'+tile+'_' in k]
                tile_filtered =  [k for k in tile_filtered if '._' not in k]
                stacked = np.empty((5, 2048, 2048))
                for n,image_file in enumerate(sorted(tile_filtered)):
                    image_int = tifffile.imread(join(exported_directory,image_file))
                    stacked[n] = image_int.astype('uint16')
                pixel_size = 0.1625
                metadata = {
                                'Pixels': {
                                    'PhysicalSizeX': pixel_size,
                                    'PhysicalSizeXUnit': 'µm',
                                    'PhysicalSizeY': pixel_size,
                                    'PhysicalSizeYUnit': 'µm'
                                },
                                'Plane': {
                                    'PositionX': [position[0]*pixel_size]*stacked.shape[0],
                                    'PositionY': [position[1]*pixel_size]*stacked.shape[0]
                                }

                            }
                tif.write(stacked.astype('uint16'),metadata=metadata)

import ashlar.scripts.ashlar as ashlar
import pathlib
import warnings
warnings.filterwarnings("ignore")
def ashlar_wrapper(
    files, 
    output='', 
    align_channel=1, 
    flip_x=False, 
    flip_y=True, 
    output_channels=None, 
    maximum_shift=500, 
    filter_sigma=5.0, 
    filename_format='Round{cycle}_{channel}.tif',
    pyramid=False,
    tile_size=None,
    ffp=False,
    dfp=False,
    plates=False,
    quiet=False,
    version=False):

    ashlar.configure_terminal()
    
    filepaths = files
    output_path = pathlib.Path(output)

    import warnings
    warnings.filterwarnings("ignore")   

    # make directory
    if not os.path.exists(output):
        os.makedirs(output)

    if tile_size and not pyramid:
        ashlar.print_error("--tile-size can only be used with --pyramid")
        return 1
    if tile_size is None:
        # Implement default value logic as mentioned in argparser setup above.
        tile_size = tile_size

    ffp_paths = ffp
    if ffp_paths:
        if len(ffp_paths) not in (0, 1, len(filepaths)):
            ashlar.print_error(
                "Wrong number of flat-field profiles. Must be 1, or {}"
                " (number of input files)".format(len(filepaths))
            )
            return 1
        if len(ffp_paths) == 1:
            ffp_paths = ffp_paths * len(filepaths)

    dfp_paths = dfp
    if dfp_paths:
        if len(dfp_paths) not in (0, 1, len(filepaths)):
            ashlar.print_error(
                "Wrong number of dark-field profiles. Must be 1, or {}"
                " (number of input files)".format(len(filepaths))
            )
            return 1
        if len(dfp_paths) == 1:
            dfp_paths = dfp_paths * len(filepaths)

    aligner_args = {}
    aligner_args['channel'] = align_channel
    aligner_args['verbose'] = not quiet
    aligner_args['max_shift'] = maximum_shift
    aligner_args['filter_sigma'] = filter_sigma

    mosaic_args = {}
    if output_channels:
        mosaic_args['channels'] = output_channels
    if pyramid:
        mosaic_args['tile_size'] = tile_size
    if quiet is False:
        mosaic_args['verbose'] = True

    try:
        if plates:
            return ashlar.process_plates(
                filepaths, output_path, filename_format, flip_x,
                flip_y, ffp_paths, dfp_paths, aligner_args, mosaic_args,
                pyramid, quiet
            )
        else:
            mosaic_path_format = str(output_path / filename_format)
            return ashlar.process_single(
                filepaths, mosaic_path_format, flip_x, flip_y,
                ffp_paths, dfp_paths, aligner_args, mosaic_args, pyramid,
                quiet
            )
    except ashlar.ProcessingError as e:
        ashlar.print_error(str(e))
        return 1

def reshape_split(image: np.ndarray, kernel_size: tuple):
        
    img_height, img_width = image.shape
    tile_height, tile_width = kernel_size
    
    tiled_array = image.reshape(img_height // tile_height, 
                               tile_height, 
                               img_width // tile_width, 
                               tile_width)
    
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array

def tile_stitched_images(image_path,outpath, tile_dim=2000):
    """
    used to tile stitched images
    
    input the directory to the files that you want to tile. 
    
    """
    if not os.path.exists(outpath):
            os.makedirs(outpath)
            
    images = os.listdir(image_path)
    images =  [k for k in images if '._' not in k]
    
    for image_file in images:
        try: 
            image = tifffile.imread(image_path +'/'+ image_file)
            
            cycle = ''.join(filter(str.isdigit, image_file.split('_')[0]))
            channel = ''.join(filter(str.isdigit, image_file.split('_')[1]))
            print('tiling: ' + image_file)
            
            image_pad = cv2.copyMakeBorder( image, top = 0, bottom =math.ceil(image.shape[0]/tile_dim)*tile_dim-image.shape[0], left =0, right = math.ceil(image.shape[1]/tile_dim)*tile_dim-image.shape[1], borderType = cv2.BORDER_CONSTANT)
            image_split = reshape_split(image_pad,(tile_dim,tile_dim))
            nrows, ncols, dim1, dim2 = image_split.shape
            x = []
            y = []
            directory = outpath +'/'+'Base_'+str(int(cycle)+1)+'_stitched-'+str(int(channel)+1) 
            if not os.path.exists(directory):
                os.makedirs(directory) 
            count = 0
            for i in range(nrows):
                for j in range(ncols):
                    count = count+1                
                    x.append(j*tile_dim)
                    y.append(i*tile_dim)
                    
                    tifffile.imwrite(directory + '/' +'tile'+str(count)+'.tif',image_split[i][j])
        except KeyError:
            continue
                
    tile_pos = pd.DataFrame()
    tile_pos['x'] = x
    tile_pos['y'] = y

    tile_pos.to_csv(outpath+'/'+'tilepos.csv', header=False, index=False)
    return
def preprocessing_main_leica(input_dirs, 
                            output_location,
                            regions_to_process = 2, 
                            align_channel = 4, 
                            tile_dimension = 6000, 
                            mip = True):

    import ISS_processing.preprocessing as preprocessing
    import os
    import pandas as pd
    
    if mip == True:
        preprocessing.leica_mipping(input_dirs=input_dirs, output_dir_prefix = output_location)
    else: 
        print('not mipping')
        
    if regions_to_process > 1:
        for i in range(regions_to_process):
            path = output_location +'_R'+str(i+1)
            
            # create leica OME_tiffs
            leica_OME_tiff(directory_base = path+'/preprocessing/mipped/', 
                                            output_directory = path+'/preprocessing/OME_tiffs/')
            
            # align and stitch images
            OME_tiffs = os.listdir(path+'/preprocessing/OME_tiffs/')
            ashlar_wrapper(files = OME_tiffs, 
                                            output = path+'/preprocessing/stitched/', 
                                            align_channel=align_channel)
            
            # retile stitched images
            tile_stitched_images(image_path = path+'/preprocessing/stitched/',
                                    outpath = path+'/preprocessing/ReslicedTiles/', 
                                    tile_dim=tile_dimension)

    return





