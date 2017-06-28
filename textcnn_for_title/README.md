# TextCNN README
Author: Kingsley Kuan

## Requirements
* tensorflow-gpu
* gensim
* regex

## Instructions

Preprocess raw titles.

```
python3 preprocess_title.py \
--input_file=<file> \
--output_file=<file>
```

Convert titles to word vector embeddings and write to TFRecord format (requires gensim and trained word2vec model).
A pretrained word2vec model, GoogleNews-vectors-negative300.bin can be obtained from <https://code.google.com/archive/p/word2vec/>

```
python3 title_to_tfrecord.py \
--word2vec_model_file=<file> \
--video_titles_file=<file> \
--video_labels_file=<file> \
--tfrecord_file=<file>
```

Use train.py to train model.

```
python3 train.py \
--tfrecord_file=<file> \
--train_dir=<dir> \
--batch_size=512 \
--num_epochs=5
```

Generate predictions as well as features from second last layer.

```
python3 generate_predictions.py \
--tfrecord_file=<file> \
--train_dir=<dir> \
--predictions_file=<file> \
--features_file=<file> \
--batch_size=1024
```


## Optional: Filter labels from titles

```
python3 filter_titles.py \
--input_file=<file> \
--output_file=<file> \
--video_labels_file=<file> \
--label_names_file=<file>
```
