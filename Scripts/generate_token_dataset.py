import os
import csv
import json
import pathlib
import argparse

def load_data(json_fpath):
    with open(json_fpath, "r") as json_f:
        data_dict = json.load(json_f)

    return data_dict

def main():
    parser = argparse.ArgumentParser(
        description="Generates a dataset of tokens from text (Tokenization).")

    parser.add_argument(
        "--dest-path",
        help="Destination output path for dataset json files.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--dictionary-path",
        help="File path to Dictionary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--text-dataset-path",
        help="File path to text Dataset.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    dest_path = args["dest_path"]
    dictionary_path = args["dictionary_path"]
    text_dataset_path = args["text_dataset_path"]

    dictionary_dict = load_data(json_fpath=dictionary_path)
    text_dataset_dict = load_data(json_fpath=text_dataset_path)

    categories = text_dataset_dict["categories"]
    categories_fpath = os.path.join(dest_path, "categories.json")
    try:
        with open(categories_fpath, "w") as json_f:
            json.dump({"data": categories}, json_f)

        print(f"Saved categories JSON files.")
    except Exception as e:
        raise e

    all_fpaths = []
    folder_index = 0

    for file_index, (person_name, person_data) in enumerate(text_dataset_dict["data"].items()):
        content_tokens = []
        split_contents = person_data["content"].split(" ")
        for index, word in enumerate(split_contents):
            if word in dictionary_dict:
                token = dictionary_dict[word]
                content_tokens.append(token)
            else:
                characters = list(word)
                token = [dictionary_dict[character] for character in characters]
                content_tokens.extend(token)

            token = dictionary_dict[" "]

            if index < len(split_contents) - 1:
                content_tokens.append(token)

        context_tokens = {}
        for category in categories:
            context_tokens[category] = {}

            contexts = person_data["context"][category]
            for context_type, context_data in contexts.items():
                split_contexts = context_data.split(" ")

                temp_context_tokens = []
                for index, word in enumerate(split_contexts):
                    if word in dictionary_dict:
                        token = dictionary_dict[word]
                        temp_context_tokens.append(token)
                    else:
                        characters = list(word)

                        token = [dictionary_dict[character] for character in characters]
                        temp_context_tokens.extend(token)

                    if index < len(split_contexts) - 1:
                        token = dictionary_dict[" "]
                        temp_context_tokens.append(token)

                context_tokens[category][context_type] = temp_context_tokens

        temp_data_dict = {
            "content": content_tokens,
            "context": context_tokens}

        curr_dir_path = os.path.join(dest_path, str(folder_index))
        os.makedirs(curr_dir_path, exist_ok=True)

        curr_file_path = os.path.join(curr_dir_path, person_name + ".json")

        all_fpaths.append(curr_file_path)
        try:
            with open(curr_file_path, "w") as json_f:
                json.dump(temp_data_dict, json_f)

            print(f"Saved {curr_file_path} file.")
        except Exception as e:
            raise e

        if file_index % 1_000 == 0 and file_index > 0:
            folder_index += 1

    file_list_fpath = os.path.join(dest_path, "DatasetList.csv")
    try:
        with open(file_list_fpath, "w") as csv_f:
            list_writer = csv.writer(csv_f, delimiter='\n')
            list_writer.writerows([all_fpaths])

        print(f"Saved filepaths CSV files.")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
