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
- npz file of predicted action, together with a txt file compatible with the annotation tool and an xls file with some statistrics
-video (optional)


