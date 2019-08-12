# deep_learning
using deep learning techniques to manipulate varied training sets 



This file contains instructions to setup environment for the practicals. We recommend using pre-made packages below if you are using Windows 10
Pre-made packages for Windows 10

For easiest installation, download this archive file and extract it to location of your choosing (approx. 3GB in size after extraction). Then launch extracted start_uefdrl19.bat. This should open a command prompt with environment activated (text (userdrl19) in the beginning of prompt).

To test if the environment works correctly, launch ipython in the prompt you opened and then command import numpy, tensorflow, keras, gym, vizdoom . If there are no errors, then environment works (don’t mind any FutureWarnings it might print).
Manual installation
Installing miniconda

Following the instruction and install Miniconda from this link: https://conda.io/miniconda.html
On Windows: Launch “Anaconda Prompt”.
Create the environment

Download environment file here, and then command

    conda env create -f=environment.yml

You also need to install ViZDoom (Note: You have to activate environment first):

    Linux: Install Linux dependencies and then install ViZDoom with pip install vizdoom
    MacOS: Install MacOS dependencies and then install ViZdoom with pip install vizdoom
    Windows: Download precompiled binaries, and extract the contents into [path_to_environment]/Lib/site-packages

Using installed environment

Activating and deactivating our environment:
For Windows

    activate uefdrl19

    deactivate uefdrl19

For Linux/Mac

    source activate uefdrl19

    source deactivate

Listing installed packages:

    conda list

Delete environment

    conda remove --name uefdrl19 --all

More tutorials for Windows users

https://conda.io/docs/user-guide/install/windows.html#install-win-silent
