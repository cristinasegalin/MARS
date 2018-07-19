## AMT_annotation
This folder contains all you need to collect, extract, process frames and export information from annotation

1. extracts frames from videos you want to annotate
2. annotate the frames with your favorite tool
4. if you used AMT you can modify `amt2csv.py`to  exports AMT annotation to csv file correcting them if the sides or ears are flipped
5. `csv2dict.py` extracts info from csv created exporting the annotations  and save them as a structure.
6. If you want to check the annotation and run some statitics use `plots_annotations.py`
7. `dict2tfrecords.py` convert the structure to tfrecords compatible format.


##                    FILES INCLUDED IN THIS FOLDER

|Name |                                Description|
|-----|--------------------------------------------|
|amt2csv_basic.py|                          extract AMT annotation and save in csv format|
|amt2csv_front.py|                          extract AMT annotation and save in csv format for front view|
|amt2csv_top.py|                            extract AMT annotation and save in csv format for top view|
|amt_error.py|                              compute performance of AMT workers|
|create_tfrecords.py|                       utility to convert dictionary into tf format|
|csv2dict_front.py|                         convert csv to dictionary for front view|
|csv2dict_miniscope_front.py|               convert csv to dictionary for front minisciope videos|
|csv2dict_miniscope_top.py|                 convert csv to dictionary for top miniscope videos|
|csv2dict_top.py|                           convert csv to dictionary for top videos|
|dict2tfrecords.py|                         convert structure to tf records|
|dict2tfrecords_front_allset.py|            convert structure to tf records for front videos both with cable and no cable|
|dict2tfrecords_front_allset_separate.py|   convert structure to tf records for front videos both with cable and no cable separating each mouse |
|dict2tfrecords_top_allset.py|              convert structure to tf records for top videos both with cable and no cable|
|dict2tfrecords_top_allset_separate.py|     convert structure to tf records for top videos both with cable and no cable separating each mouse |
|plot_annotations.py|                       utility to check and plot annotations
