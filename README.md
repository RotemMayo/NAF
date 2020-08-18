# NAF
This is not the original NAF Repo! All credit goes to https://github.com/CW-Huang.


## Original docs:
Experiments for the Neural Autoregressive Flows paper

This repo depends on another library for pytorch modules: https://github.com/CW-Huang/torchkit

To download datasets, please modify L21-24 of `download_datasets.py`. 

## Troubleshooting:
The datasets might contain nan values causing losses not to work. In order to solve this use np.nan_to_num on the data when loading.
