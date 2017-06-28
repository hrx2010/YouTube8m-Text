#!/usr/bin/env python3

# Author: Kingsley Kuan

import argparse
import regex as re
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocesses video title')

    parser.add_argument('--input_file',
                        type=str,
                        default='data_train/video_titles_raw.json',
                        help='JSON file to read raw video titles')

    parser.add_argument('--output_file',
                        type=str,
                        default='data_train/video_titles.json',
                        help='JSON file to write preprocessed video titles to')

    args = parser.parse_args()
    return args

def extract_video_titles(input_file, output_file):
    with open(input_file, encoding='utf-8') as file:
        titles = json.load(file)

    count = 0

    for video_id, title in titles.items():
        # Remove all punctuation and symbols in unicode
        title = re.sub(r'[\p{P}\p{S}]+', '', title)
        titles[video_id] = title.split(' ')

        count += 1
        print('{}: {}'.format(count, video_id), end='\r', flush=True)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(titles, file)

if __name__ == '__main__':
    args = parse_args()
    extract_video_titles(args.input_file, args.output_file)
