# Computer-vision - Final Project
## Goal
Suggest a modification to MSTCN++ aiming to improve performances of gesture recognition task over specific data of physicians performing practice surgury operations.<br>

## Description
<ul>
  <li> Used MSTCN++ pytorch implementation (https://github.com/sj-li/MS-TCN2) of the paper [MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/2006.09220.pdf) to perform gesture recognition over data of physicians performing surgical activity as baseline results. </li>
  <li> Implementation of modifications to the original architecture such as weighted loss, adding GRU stages and performing down sampling.</li>
  <li> Reporting evaluation metrics.</li>
  <li> Tagging videos.</li>
 </ul>

## How to run?
#### Prepare envoriment
1. Clone this project
2. pip/conda install the requirments.txt file

### Reproduce results
#### main.py
The file which does all of the heavy lifting is `main.py`. <br>
`main.py` is responsible for running the baseline and modified architectures , performing gesture recognition over the data videos. It also reports loss and accuracy over train and validation sets and generates graphs in clearML. <br>

The script assumes labels are provided in a directory called transcriptions_gestures where each video has a corresponding text file with the same name as the video, holding the ground truth labels in the following frame format:

where `videos` is a directory in the same dir as `main.py` and it contains video files. In the `videos` directory there are two other directories called `tools_left` and `tools_right`, each contaning a corrosponding text file with the same name as a video in `videos` dir and the text files contain the labels in frame format:
```
0 524 G0
525 662 G1
663 808 G2
809 898 G3
899 970 G4
```
it can be run using the following commands:

1. To get baseline results run:
```
main.py --action baseline 
```
2. To train the new architecture (our modification) run:
```
main.py --action train
```
3. To run the tradeoff experiment run:
```
main.py --action train_tradeoff
```
All the above commands include performing prediction over the test data

```
