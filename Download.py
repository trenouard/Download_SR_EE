import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.ion()
import pandas as pd

from datetime import datetime
from utils import SDS_download, SDS_preprocess, SDS_process, SDS_tools

# region of interest (longitude, latitude)

        ### SENNAR AOI ### 
polygon = [[[33.428558, 13.696721],
            [33.428558, 13.617978],
            [33.506682, 13.696721],
            [33.506682, 13.617978],
            [33.428558, 13.696721]]] 


#polygon = [[[33.407151, 13.902460],
#            [33.407151, 13.580285],
#            [33.936717, 13.902460],
#            [33.936717, 13.580285],
#            [33.407151, 13.902460]]] 

# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = SDS_tools.smallest_rectangle(polygon)
# date range
dates = ['2018-01-01', '2022-04-01']
# satellite missions
sat_list = ['S2', 'L8'] #['L5', 'L7' ]
# name of the site
sitename = 'Sennar_small'
# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'data')
# put all the inputs into a dictionnary
inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list, 'sitename': sitename, 'filepath':filepath}

# before downloading the images, check how many images are available for your inputs
SDS_download.check_images_available(inputs);

# inputs['include_T2'] = True
### comment the next line if the Surface Reflectances have already been downloaded
metadata = SDS_download.retrieve_images(inputs)

metadata = SDS_download.get_metadata(inputs) 

settings = { 
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'cloud_mask_issue': False,  # switch this parameter to True if pixels are masked (in black) on many images  
    # add the inputs defined previously
    'inputs': inputs
}
    

SDS_process.get_ndvi_lst(metadata, settings)