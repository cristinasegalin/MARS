## A_annotation
This folder contains all you need to collect, extract, process frames and export information from annotation

1. extracts frames from videos you want to annotate
2. annotate the frames with your favorite tool
4. if you used AMT you can modify `amt2csv.py`to  exports AMT annotation to csv file correcting them if the sides or ears are flipped
5. `csv2dict.py` extracts info from csv created exporting the annotations  and save them as a structure.
6. If you want to check the annotation and run some statitics use `plots_annotations.py`
7. `dict2tfrecords.py` convert the structure to tfrecords compatible format.
