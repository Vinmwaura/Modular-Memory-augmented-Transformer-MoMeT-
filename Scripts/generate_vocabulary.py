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
        description="Generate vocabulary using characters (ASCII printable) and words (Delineated by spaces).")

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
    locations_list = load_data(csv_fpath=locations_list_path)
    sentences_list.extend(locations_list)

    raw_template_path = os.path.join(template_path, "raw_template_text.txt")
    raw_template_list = load_data(csv_fpath=raw_template_path)
    sentences_list.extend(raw_template_list)

    # Combine all lists into one.
    all_lists = person_likes_list + person_dislikes_list + person_hobbies_list + universities_list + locations_list + raw_template_list

    # Get every unique character from the list of sentences.
    unique_characters = set()
    for words in all_lists:
        for character in words:
            unique_characters.add(character)

    # Hack: Ensure all lowercase and uppercase ASCII characters are represented.
    for ascii_lowercase in string.ascii_lowercase:
        unique_characters.add(ascii_lowercase)
    for ascii_uppercase in string.ascii_uppercase:
        unique_characters.add(ascii_uppercase)

    unique_characters_list = list(unique_characters)
    unique_characters_list.sort()

    all_words = unique_characters_list
    for sentence in sentences_list:
        words_split = sentence.split(" ")
        all_words.extend(words_split)

    unique_words = list(set(filter(None, all_words)))
    unique_words.sort()

    vocabulary_data = {"tokens_to_id": {}}
    for index, unique_word in enumerate(unique_words):
        vocabulary_data["tokens_to_id"][unique_word] = index

    """
    Special Tokens used in delineating the start and end of specific
    information as well as pad input.
    """
    len_vocabulary = len(vocabulary_data["tokens_to_id"])
    vocabulary_data["special_tokens_to_id"] = {
        "pad_token": len_vocabulary + 0,
        "start_prompt": len_vocabulary + 1,
        "end_prompt": len_vocabulary + 2,
        "start_tag": len_vocabulary + 3,
        "end_tag": len_vocabulary + 4,
        "SContext": len_vocabulary + 5,
        "EContext": len_vocabulary + 6,
        "start_response": len_vocabulary + 7,
        "end_response": len_vocabulary + 8}

    try:
        vocabulary_fpath = os.path.join(dest_path, "Vocabulary.json")
        with open(vocabulary_fpath, "w") as json_f:
            json.dump(vocabulary_data, json_f, indent=4)

        print("Successfully saved vocabulary!")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
