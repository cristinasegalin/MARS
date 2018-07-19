# MARS

This is the first version of MARS software.

### Requirements:
- Tensorflow-1.1-rc1 ([how to](http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html))
- cuda 8.0
- cudNN 5.1
- pyside
- [models](https://www.dropbox.com/sh/sw516fmrs6rmhgy/AAB4Ljg8qnUmrfGm5GKCNSKDa?dl=0)

### How to run with interface:
- Open a terminal and navigate to the location where the folder was downloaded
- use the command `python MARS.py` to launch the software
- select the folder contaning the seq videos
- select what to do 
- push RUN MARS button

The software will save the results in an `output` folder in the same folder where videos are.
The outputs are:
- json files of top and front pose
- json and mat files of extracted features 
- json file of predicted action, together with a txt file compatible with the annotation tool and an xls file with some statistrics
-video (optional)


