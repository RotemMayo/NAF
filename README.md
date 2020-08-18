# NAF
This is not the original NAF Repo! All credit goes to https://github.com/CW-Huang.


## Original docs:
Experiments for the Neural Autoregressive Flows paper

This repo depends on another library for pytorch modules: https://github.com/CW-Huang/torchkit

To download datasets, please modify L21-24 of `download_datasets.py`. 


## Installation 
In terminal run:

    conda env create -f NAF_conda_env.yml

    conda activate NAF


 Sanity check:

    which pip

 Should return a path that includes envs/NAF


In a separate directory (or in a subdirectory of this project):

        git clone https://github.com/CW-Huang/torchkit.git
        cd torchkit
        pip install -e .
        
To check if it worked perform:

        conda list
        
torchkit should appear amongst the installed packages.

## Congratulations

Thats it you're done! When running the files make sure to activate the env before hand by doing:

    conda activate NAF


If you wish to have a different name for the venv simply change the name in the first line of the file.



## Troubleshooting:
The datasets might contain nan values causing losses not to work. In order to solve this use np.nan_to_num on the data when loading.
