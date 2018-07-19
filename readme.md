# MARS - Mouse Action Recognition System

In this repoistory we present MARS, a mouse tracker and behavior classifier tool.
The system pipeline follow:


![pipeline](https://github.com/cristinasegalin/MARS/blob/master/pipeline.png)

Here you can find two folders:
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

