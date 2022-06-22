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
