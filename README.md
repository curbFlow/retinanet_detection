# Vehicle Detection with RetinaNet

We use this repo to train and evaluate a Keras Retinanet (ResNet18 backbone) Object Detection Model.

## Dataset 
This repo consumes datasets in the CSV format.  
Each row of the csv contains a bounding box of the form : imagePath, xmin, ymin, xmax, ymax, labelname.
An additional classes.csv file is also used to learn the index mapping from labelname to class indexes.  
Each row is of the form: [labelname, index]
##Model training.
To train the model, run   
` python3 train_resnet18.py --train_path train.csv  --val_path val.csv --classes_path categories.csv --fsize 128
`  
train_path = Path to train csv.  
val_path = Path to val csv.  
classes_path = Path to classes csv.
## Model evalulation
To evaluate the model, run 
`python3 eval.py`

##Requirements
This requires TF=2.1.0 and Keras=2.3.1

