#!/usr/bin/env python3

# Author: Kingsley Kuan

import argparse
import json
import csv
import regex as re

def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter out labels from video titles')

    parser.add_argument('--input_file',
                        type=str,
                        default='data_train/video_titles.json',
                        help='JSON file containing preprocessed video titles')

    parser.add_argument('--output_file',
                        type=str,
                        default='data_train/video_titles_filtered.json',
                        help='JSON file to write filtered video titles to')

    parser.add_argument('--video_labels_file',
                        type=str,
                        default='labels.json',
                        help='JSON file containing the labels for each video')

    parser.add_argument('--label_names_file',
                        type=str,
                        default='label_names.csv',
                        help='CSV file containing mapping from label to name')

    args = parser.parse_args()
    return args

def filter_titles(input_file, output_file, video_labels_file, label_names_file):
    with open(input_file, encoding='utf-8') as file:
        titles = json.load(file)

    with open(video_labels_file, encoding='utf-8') as file:
        video_labels = json.load(file)

    # Read in mapping from label id to label name
    label_names = {}
    with open(label_names_file, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label_id = int(row['label_id'])
            label_name = row['label_name']

            # Remove all punctuation and symbols in unicode
            label_name = re.sub(r'[\p{P}\p{S}]+', '', label_name)
            label_names[label_id] = label_name.lower()

    match = 0
    no_match = 0

    for video_id in titles:
        # Join list of words into title
        title = ' '.join(titles[video_id])

        # Get video labels and map from ids to names
        labels = video_labels[video_id]
        labels = [label_names[label_id] for label_id in labels]

        # Use regex to find and remove labels inside video title
        for label in labels:
            labels_regex = r'\b' + label + r'\b'
            labels_regex = re.compile(labels_regex, flags=re.IGNORECASE)

            if labels_regex.search(title) is not None:
                match += 1
            else:
                no_match += 1

            print("Match: {} | No Match: {}".format(match, no_match),
                  end='\r', flush=True)

            title = labels_regex.sub('', title)

        titles[video_id] = title.split()

    print("Match: {} | No Match: {}".format(match, no_match))

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(titles, file)

if __name__ == '__main__':
    args = parse_args()
    filter_titles(args.input_file,
                  args.output_file,
                  args.video_labels_file,
                  args.label_names_file)
