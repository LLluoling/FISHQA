## This is the simplified version of FISHQA model !

# Paper web
paper: https://www.ijcai.org/proceedings/2018/0590.pdf


# Code Introduction 

 1. preprocess_train.py / preprocess_test.py
 Preprocess training dataset/test dataset, remember to modify the dictionary, fiterwords based on your own datasets
 2. train.py
Set params based on your own datasets and train you own model
3. test.py 
4. view.py
For simple attention visualization 

# Data Introduction 
1. Modify your own queries(FISHQA/Query) based on your own datasets and prior knowledge.
2. As the our dataset is private, we cannot release it. We put two raw samples in file train_data and test_data individually.
