# Download_SR_EE
Download Landsat and Sentinel Surface Reflectance products from GEE

### Description

### Table of Contents

* [1. Installation](#chapter1)
    * [1.0 Clone the repository from GitHub](#section_1_0)
    * [1.1 Create an environment with Anaconda](#section_1_1)
    * [1.2 Activate Google Earth Engine Python API](#section_1_2)
* [2. Usage](#chapter2)
    * [2.1 Retrieval of the satellite images](#section_2_1)

## 1. Installation <a class="anchor" id="chapter1"></a>

### 1.0 Clone the repository from GitHub <a class="anchor" id="section_1_0"></a>

Start by installing **GitHub desktop** on your machine.

Then open GitHub desktop and follow this [tutorial](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop) to clone the repository on your machine.

### 1.1 Create an environment with Anaconda <a class="anchor" id="section_1_1"></a>

You first need to install the required Python libraries in an environment to run this package. To do this, we will use Anaconda, which can be downloaded freely [here](https://www.anaconda.com/products/individual).

Once you have it installed, open the **Anaconda PowerShell** prompt (a terminal window in MacOS and Linux) and use `cd` (change directory) and `ls` (list files and directories) commands to go to the folder where you have downloaded the **Download_SR_EE** repository.

Create a new environment named `download_sr_ee` that will contain all the required packages:

`conda env create -f environment.yml -n download_sr_ee`

Now, activate the environment:

`conda activate download_sr_ee`

To confirm that you have successfully activated download_sr_ee, your command line prompt should begin with *download_sr_ee*.
 
### 1.2 Activate Google Earth Engine Python API <a class="anchor" id="section_1_2"></a>

You first need to sign up to Google Earth Engine at https://signup.earthengine.google.com/. 

Once your request has been approved, with the `coastpred` environment activated, run the following command on the Anaconda Prompt to link your environment to the GEE server.

`earthengine authenticate`

A web browser will open, login with a gmail account and accept the terms and conditions. Then copy the authorization code into the Anaconda terminal.

Now you are ready to start using the `Download_SR_EE` toolbox!

*Note: remember to always activate the environment with `conda activate download_sr_ee` each time you are preparing to use the package.*

## 2. Usage <a class="anchor" id="chapter2"></a> 

An example of how to run the software in a Jupyter Notebook is provided in the repository (`Download.ipynb`). To run this, first activate your `download_sr_ee` environment with `conda activate download_sr_ee` (if not already active), and then type:

`jupyter notebook`

A web browser window will open. Point to the directory where you downloaded this repository and click on `Download.ipynb`.

The following sections guide the reader through the different functionalities of `Download_SR_EE` with an example at Sennar, Sudan. If you prefer to use Spyder, PyCharm or other integrated development environments (IDEs), a Python script named Download.py is also included in the repository.

A Jupyter Notebook combines formatted text and code. To run the code, place your cursor inside one of the code sections and click on the `run cell` button (or press `Shift` + `Enter`) and progress forward.

### 2.1 Retrieval of the satellite images <a class="anchor" id="section_2_1"></a>

To retrieve from the GEE server the available satellite images cropped around the user-defined region of coastline for the particular time period of interest, the following variables are required:

* `polygon`: the coordinates of the region of interest (longitude/latitude pairs in WGS84)
* `dates`: dates over which the images will be retrieved (e.g., dates = ['2017-12-01', '2018-01-01'])
* `sat_list`: satellite missions to consider (e.g., sat_list = ['L5', 'L7', 'L8', 'S2'] for Landsat 5, 7, 8 and Sentinel-2 collections)
* `sitename`: name of the site (this is the name of the subfolder where the images and other accompanying files will be stored)
* `filepath`: filepath to the directory where the data will be stored

The call `metadata = SDS_download.retrieve_images(inputs)` will launch the retrieval of the images and store them as .TIF files (under /filepath/sitename). The metadata contains the exact time of acquisition (in UTC time) of each image, its projection and its geometric accuracy. If the images have already been downloaded previously and the user only wants to run the NDVI/LST processing, the metadata can be loaded directly by running `metadata = SDS_download.get_metadata(inputs)`.

The cell below shows an example of inputs that will retrieve all the images of Sennar, Sudand acquired by Sentinel-2 in 2015.


```python
# region of interest (longitude, latitude)
polygon = [[[33.428558, 13.696721],
            [33.428558, 13.617978],
            [33.506682, 13.696721],
            [33.506682, 13.617978],
            [33.428558, 13.696721]]] 
# it's recommended to convert the polygon to the smallest rectangle (sides parallel to coordinate axes)       
polygon = SDS_tools.smallest_rectangle(polygon)
# date range
dates = ['2015-12-01', '2015-12-31']
# satellite missions
sat_list = ['S2']
# name of the site
sitename = 'Sennar'
# directory where the data will be stored
filepath = os.path.join(os.getcwd(), 'data')
```

**Note:** The area of the polygon should not exceed 100 km2, so for very large ROI split it into multiple smaller polygons.

### 2.2 NDVI/LST computation <a class="anchor" id="section_2_2"></a>

To compute NDVI and LST, the following user-defined settings are needed:

* `cloud_thresh`: threshold on maximum cloud cover that is acceptable on the images (value between 0 and 1 - this may require some initial experimentation).
* `cloud_mask_issue`: switch this parameter to True if pixels are masked (in black) on many images

An example of settings is provided here:


```python
settings = { 
    # general parameters:
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover
    'cloud_mask_issue': False,  # switch this parameter to True if pixels are masked (in black) on many images  
    # add the inputs defined previously
    'inputs': inputs
}
```

Once all the settings have been defined, the computation can be launched by calling:

`SDS_process.get_ndvi_lst(metadata, settings)`

It will store the NDVI and LST as .TIF files and .ZARR files (under /filepath/sitename). 
