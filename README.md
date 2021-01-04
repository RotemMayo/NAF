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

Project dir on cluster:

        /usr/people/snirgaz/rotemov/NAF

To run with my settings simply use the commands:

        cd <project_directory>
        sbatch <sh script> <input args for some of the scripts>

Read the scripts before you use them to understand if they require input args. If they do it is recommended to wrap 
each arg with quotation marks.

You might need to change the user variable for scripts which are not yonitq.

Simply switch the line:

        #SBATCH -A rotemov-account
        
To:

        #SBATCH -A <user-name>-account

Alternatively you can duplicate and modify mine to create running scripts of your own.

In order to check jobs you ran simply use the command:

        squeue --user=<user name>

In order to check jobs running on yonitq use the command:

        squeue --partition=yonitq

Logs are stored in logs folder with the format:

        <job name>_<job id>.<log/out>

Simply look for the one with your job id as it is the only unique feature.


## Troubleshooting

The datasets might contain nan values causing losses not to work. In order to solve this use np.nan_to_num on the data when loadingg.sg   lakfg.g.g.g

## TODO
1. Smaller bin sizes
2. Train on removed events
3. tsne without loss
4. train on mjj translation + 1 obs
5. 2 of papers sent by Tau (review on ML and embedding clustering)
6. AE filtering
7. Anomally score slide (need to ask tau to send)
8. https://pypi.org/project/kmeans-pytorch/
