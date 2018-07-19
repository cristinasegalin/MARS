# MARS

### Requirements:
- Tensorflow-1.1-rc1 ([how to](http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html))
- cuda 8.0
- cudNN 5.1
- pyside
- [models](http://www.vision.caltech.edu/~segalinc/git_data/gui/v1_7.zip)

### How to run with interface:
- Open a terminal and navigate to the location where the folder was downloaded
- double click on MARS icon
- select the folder contaning the seq videos
- select what to do 
- push RUN MARS button

The software will save the results in an `output` folder in the same folder where videos are.
The outputs are:
- json files of top and front pose
- npz and mat files of extracted features
- npz, mat file of predicted action, together with a txt file compatible with the annotation tool
- mp4 video of result (optional)

### Instruction to set up launchers:

Change the absolute path where the MARS_V1_7 folder is in the following .desktop and .sh files:
- MARS.desktop and MARS.sh
- MARS_rename_version_utility.desktop
- MARS_dump_bento_dir.desktop

Be sure that in MARS.sh the export path to the where CUDA is installed is correct. You can find it usually in /home/$USER/.bashrc



