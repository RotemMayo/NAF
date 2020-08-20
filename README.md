# NAF ML4Jets HUJI

This is not the original NAF Repo! All credit for the orignal architecture goes to https://github.com/CW-Huang.

## Original docs

Experiments for the Neural Autoregressive Flows paper: https://arxiv.org/abs/1804.00779

This repo depends on another library for pytorch modules: https://github.com/CW-Huang/torchkit

To download datasets, please modify L21-24 of `download_datasets.py`. 

## Compatability

This was checked on several Linux OS (Ubuntu, gentoo, Debians), MAC compatibility check to come.

For downloading the datasets you will need bash.

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


To download the dataset go to the project directory (NAF):

        cd external_maf/datasets/data/lhc
        sh lhc_download.sh
        
Thats it you're done!

## Running

When running the files make sure to activate the env and updating the repository:

        cd <project_directory>
        git pull
        conda activate NAF
        python maf_experiments

If you wish to have a different name for the venv simply change the name in the first line of the file.

## Running on cluster

Connect to landau cluster using (if not on a HUJI netowrk use SambaVPN: https://ca.huji.ac.il/samba):

       ssh <username>@landau.fh.huji.ac.il

Simply run the commands:

        cd <project_directory>
        sbatch --export=dataset='lhc' cluster_runner.sh

cluster_runner.sh might need to adjusted to your own user. Simply switch the line:

        #SBATCH -A rotemov-account
        
To:

        #SBATCH -A <user-name>-account

## Troubleshooting

The datasets might contain nan values causing losses not to work. In order to solve this use np.nan_to_num on the data when loading.
