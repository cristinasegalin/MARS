## BEHAVIOR CLASSIFIER

In this folder are all files used to train behavior classifier.

We train different classifiers using xgboost or sklearn-MLPClassifier. We then use a moving average smoothing and hmm to
smooth the prediction.
We train classifiers for `close investigation, mount, attack`  and `interaction` behaviors.
We tested on the rae features and on windoed features computer with a given radius around a frame.

## IN THIS FOLDER

|Filename | Description|
|---------|------------|
|
