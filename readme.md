# MARS - Mouse Action Recognition System

In this repoistory we present MARS, a mouse tracker and behavior classifier tool.
Dataset, more info and documentation can be found on [MARS website](http://www.vision.caltech.edu/~segalinc)

The system pipeline is composed by 2 mouse detectors, a pose detector, features extraction and behavior classifiers.

![pipeline](https://github.com/cristinasegalin/MARS/blob/master/pipeline.png)

## In this repo:

- **develop** : all you need to build the pipepline step by step, from the annotation, detection, pose, behavior classifiers
and export the models.
- **gui**: the MARS gui software ready to to use.

## Prerequisites

- python2.7 (windows user, coming soon!)
- tensorflow 1.1

### Supported video format

- seq (can be read with [this](https://github.com/cristinasegalin/MARS/tree/master/develop/seqIo.py) utility)
- avi
- mpg

### Behavior annotator used

- Caltech behavior annotator (part of [Piotr's Toolbox](https://pdollar.github.io/toolbox/)

