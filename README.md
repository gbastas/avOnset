# avOnset

## Set environment

conda create -n visonset python=3.6.12
conda activate visonset
pip install -r requirements.txt

## Dataset download and skeleton representations

First you need to download the whole URMP dataset (this might take some time...):

```
www2.ece.rochester.edu/projects/air/projects/URMP.html
```

Then you can download the readily extracted skeleton data from here:
```
https://imisathena-my.sharepoint.com/:u:/g/personal/g_bastas_athenarc_gr/ERD4ZZ0iPuVPkGRnIWC1qd4BPYoxCj3NSg2qJBPTUiyFBw?e=G0KzVj
```

## D.I.Y. (documentation Under Construction)


<!-- If you want to extract the skeletons yourself, you need to dowload OpenPose and run it for each multi-instrument video-performance. This is easier to achieve from Windows os. First, we download openpose from this link https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases and we run openpose/models/getModels.bat. Next we run ```openpose/bin/OpenPoseDemo.exe``` for multiple videos using the script below (run inside the ```openpose/``` dir) to get the poses in the form of json files and videos:

```
python path\to\avOnset\src\run_multiple_openpose.py --pathToData path\to\dataset --poly {True,False}
```

Then run:
```
python cop_videos.py --pathToData path/to/data
``` -->


## How to run the model

```
cd TCN/cross_val_av/
python music_test.py
```