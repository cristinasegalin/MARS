# MARS

This is the first version of MARS software.

### Requirements:
- Tensorflow-1.1-rc1 ([how to](http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html))
- cuda 8.0
- cudNN 5.1
- pyside
- [models](https://www.dropbox.com/sh/sw516fmrs6rmhgy/AAB4Ljg8qnUmrfGm5GKCNSKDa?dl=0)

### How to run in without interface (batch of videos):
- Open terminal and navigate to the location where is this code
- use the command `python batched_optimized_pipeline.py --path path/where/videosfodlers/are/ --tasks ${1 2 3 4} --ext videoext` to run the system, where `tasks` is an array in range of 4 and accepted extensions are seq, mpg, 




