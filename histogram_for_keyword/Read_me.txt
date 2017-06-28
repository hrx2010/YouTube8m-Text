This folder handles everything that concerns keywords features.

Raw keywords for training and development sets are 'input-train.txt' and 'input-dev.txt'
For validation set it is 'combined_metadata_val.json'

#HISTOGRAMS CREATION

1-Gather all the keywords, pre-process them, and cluster them into 1024 centroids via kmeans
For this, run kmeans_kw.py
The arguments we used are set as default.
This script exports a numpy array of the centroids coordinates.

2-Load the keywords, pre-process them (again), and map them to their closest centroid in order to get one histogram per video
Run histograms.py
This script needs to load a numpy array of the centroids (cf 1)
Histograms are saved

If you want to do this whole process with keywords with labels removed, run the following two scripts instead:
kmeans_no_label.py
histograms_no_label.py
Labels of each video are saved as 'labels_groundtruth_train.csv' (same file for train + dev sets) and 'labels_groundtruth_val.txt' for the validation set.
Thus, 2 different ways to read these labels in all the scripts...
By default histograms are saved in data/keywords_features_kmeans_xxx

#UNIGRAM MODEL

3-Create a probability matrix of the appereance of each label given each keyword. Number of rows/columns is equal to number of labels/keywords.
Run joint_probabilities.py.
It is used on the training set.
Creates the given matrix in .csv format
It is a big file (~5gb)

4-From that matrix, build a simple classification model. Given a video and its keywords, build a probability vector of the labels given this current set of vectors.
Since these probabilities are small, we take their log and then normalize between 0 and 1.
Run unigram_pred.py
It is used on the validation set. 
Outputs just a number (the classification GAP (with top 20 labels)). The GAP is computed thanks to another script (eval_util.py).


#DECODER MODEL

5-Now we wonder how good predictions we can get with just the keywords histograms.
Run training_keywords.py
It will train a simple multi-fully-connected-layer neural network classifier, based on model_keywords.py
A few annex scripts are used to compute the GAP, as it is the metric that we want.
The final printed output will be the GAP on all the test set.
This script lets you choose in command line:
-to use the development or validation set for testing
-number of epochs
-batch-size
WARNING: be careful to precise the right data paths when building the training and testing sets ('training_data_path' and 'testing_data_path').
You can call data using the labels (for instance 'keywords_features_kmeans_train' and 'keywords_features_kmeans_no_label_train').