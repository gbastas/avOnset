# avOnset

This is the code for the article [Convolutional Networks for Visual Onset Detection in the Context of Bowed String Instrument Performances](https://zenodo.org/record/5043899).

## Set environment

```
conda create -n visonset python=3.6.12
conda activate visonset
pip install -r requirements.txt
```
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
python crop_videos.py --pathToData path/to/data
``` -->


## Feature extraction

```
mkdir PrepdData
cd src/
python data_prep_av_bulk.py --pathToStore ../prepdData/extracted_features --pathToURMP {path/to/URMPdataset} --audio_feats melspec -HandROIs -optflow
```
If you don't want to bother with HandROIs (and their optical flow representation) don't use the last two arguments. The script will run a lot faster and the space occupied in the disc will be smaller.

## How to run the model


```
mkdir models
mkdir imgs
mkdir results
cd TCN/cross_val_av/
```


For each different input configuration, you can choose to train a TCN network (one for each cross-validation fold set if you use argument ```--monofold False```) by using the argument ```-train```. The models will be stored to ```./models``` dir.



**HandROIs**:
```
python music_test.py --epochs 100 --modality HandROIs --monofold False {-train}
```
**Visual**:
```
python music_test.py --epochs 200 --modality Visual --monofold False {-train}
```
**Insights** (Under construction) [use it after training models for the separate modalities]:
```
python music_test.py --modality Body-Hand --monofold False -multiTest
```
[Extra:] **Audio**:
```
python music_test.py --epochs 200 --modality Audio --monofold False -rescaled {-train}
```


## Demo

For audio onset detection (trained on URMP string performances), click on **Browse** button and upload a ```.wav``` file. If you click on **Submit** the audio wavefrom appears.
Then click on **Audio Onset Detect** and the predected onsets will be visualized on the waveform.

(Video demo under construction)
<!-- Upload a one-instrument recorded video performance (.mkv or .mp4). A waveform and a video should appear.
Then upload a compressed (.tar.gz) dir containing a dir named "video" that includes all the extracted skeletons from OpenPose in .json form.
Click "Audio/Video Onset Dection" to visualize predicted onsets from the audio/video-based pre-trained model. -->

https://apps.ilsp.gr:5006/onsets/