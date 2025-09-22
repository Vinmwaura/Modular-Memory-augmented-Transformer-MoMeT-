import os
import csv
import json
import string
import pathlib
import argparse

from itertools import chain

def load_data(csv_fpath, delimiter='\n'):
    with open(csv_fpath) as csv_f:
        reader = csv.reader(csv_f, delimiter=delimiter)
        data_list = list(reader)

    data_list_flat = list(chain.from_iterable(data_list))
    return data_list_flat

def main():
    parser = argparse.ArgumentParser(
        description="Generate dictionary using characters (ASCII printable) and words (Delineated by spaces).")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--lists-path",
        help="File path to CSV List.",
        required=False,
        default="./csv_List",
        type=pathlib.Path)
    parser.add_argument(
        "--template-path",
        help="File path to JSON Template.",
        required=False,
        default="./json_Template",
        type=pathlib.Path)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    list_path = args["lists_path"]
    template_path = args["template_path"]

    sentences_list = []

    likes_list_path = os.path.join(list_path, "Likes.csv")
    person_likes_list = load_data(csv_fpath=likes_list_path)
    sentences_list.extend(person_likes_list)

    dislikes_list_path = os.path.join(list_path, "Dislikes.csv")
    person_dislikes_list = load_data(csv_fpath=dislikes_list_path)
    sentences_list.extend(person_dislikes_list)

    hobbies_list_path = os.path.join(list_path, "Hobbies.csv")
    person_hobbies_list = load_data(csv_fpath=hobbies_list_path)
    sentences_list.extend(person_hobbies_list)

    universities_list_path = os.path.join(list_path, "Universities.csv")
    universities_list = load_data(csv_fpath=universities_list_path)
    sentences_list.extend(universities_list)

    locations_list_path = os.path.join(list_path, "Locations.csv")
    kenya_locations_list = load_data(csv_fpath=locations_list_path)
    sentences_list.extend(kenya_locations_list)

    raw_template_path = os.path.join(template_path, "raw_template_text.txt")
    raw_template_list = load_data(csv_fpath=raw_template_path)
    sentences_list.extend(raw_template_list)

    all_words = list(string.printable)
    for sentence in sentences_list:
        words_split = sentence.split(" ")
        all_words.extend(words_split)

    unique_words = list(set(filter(None, all_words)))
    unique_words.sort()

    dictionary_data = {}
    for index, unique_word in enumerate(unique_words):
        dictionary_data[unique_word] = index

    try:
        dictionary_fpath = os.path.join(dest_path, "Dictionary.json")
        with open(dictionary_fpath, "w") as json_f:
            json.dump(dictionary_data, json_f, indent=4)

        print("Successfully saved dictionary!")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
