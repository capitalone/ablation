#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data 
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip
mv UCI\ HAR\ Dataset/train/X_train.txt har_x_train.txt 
mv UCI\ HAR\ Dataset/train/y_train.txt har_y_train.txt 
mv UCI\ HAR\ Dataset/test/X_test.txt har_x_test.txt 
mv UCI\ HAR\ Dataset/test/y_test.txt har_y_test.txt 
rm -rf UCI\ HAR\ Dataset

wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
