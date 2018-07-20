## BEHAVIOR CLASSIFIER

In this folder are all files used to train behavior classifier.

We train different classifiers using xgboost or sklearn-MLPClassifier. We then use a moving average smoothing and hmm to
smooth the prediction.
We train classifiers for `close investigation, mount, attack`  and `interaction` behaviors.
We tested on the rae features and on windowed features computer with a given radius around a frame.
To use the script you just need to change the `video_path` path where are the videos, features and annotations, the `label2id` variable to map the nominal behaviors into labels.

Pretrained model can be found [here](http://www.vision.caltech.edu/~segalinc/git_data/gui/v1_7.zip)

## IN THIS FOLDER

|Filename | Description|
|---------|------------|
|top.py| train and test classifier on top features|
|top_interaction.py| train and test calssifier on top features for just interacion behavior|
|top_pcf.py| train and test classifier on top and front pixel change features|
|top_interaction.py train and test calssifier on top and front pixel change features for just interacion behavior|
|top_pcf_wnd.py| train and test classifier on top and front pixel change features windowed|
|top_pcf_wnd_interaction.py| train and test calssifier on top and front pixel change features windowed for just interacion behavior|
|top_wnd.py| train and test classifier on top features windowed|
|top_wnd_interaction.py| train and test calssifier on top features windowed for just interacion behavior|


## INTERACTIVE TRAINING COMING SOON!
