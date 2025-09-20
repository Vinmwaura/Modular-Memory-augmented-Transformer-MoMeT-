import csv
import json
import random

from itertools import chain

from faker import Faker
from faker.providers import DynamicProvider

def load_data(csv_fpath):
    with open(csv_fpath) as csv_f:
        reader = csv.reader(csv_f, delimiter='\n', quotechar='|')
        data_list = list(reader)

    data_list_flat = list(chain.from_iterable(data_list))
    return data_list_flat

def generate_data(data_generator, data_json):
    person_FName = data_generator.first_name()

    family_LName = data_generator.last_name()

    job = data_generator.job()
    father_FName = data_generator.first_name_male()
    mother_FName = data_generator.first_name_female()

    person_likes = data_generator.person_likes()
    person_dislikes = data_generator.person_dislikes()
    person_locations = data_generator.kenya_locations()
    person_hobbies = data_generator.person_hobbies()

    content = data_json["Content"].format(
        FName=person_FName,
        LName=family_LName,
        Occupation=job,
        Location=person_locations,
        Likes=person_likes,
        Dislikes=person_dislikes,
        Hobbies=person_hobbies,
        Mother_FName=mother_FName,
        Mother_LName=family_LName,
        Father_FName=father_FName,
        Father_LName=family_LName)

    content_fields = data_json["Content_fields"]
    temp_content = [x.lstrip() for x in content.split(";")][:-1]
    content_dict = dict(zip(content_fields, temp_content))

    context_data = {}
    keys_list = list(data_json["Context"].keys())
    for key in keys_list:
        random_item = random.choice(data_json["Context"][key])
        context_data[key] = {
            "Prompt": random_item["Prompt"].format(
                FName=person_FName,
                LName=family_LName,
                Occupation=job,
                Location=person_locations,
                Likes=person_likes,
                Dislikes=person_dislikes,
                Hobbies=person_hobbies,
                Mother_FName=mother_FName,
                Mother_LName=family_LName,
                Father_FName=father_FName,
                Father_LName=family_LName
            ),
            "Response": random.choice(random_item["Response"]).format(
                FName=person_FName,
                LName=family_LName,
                Occupation=job,
                Location=person_locations,
                Likes=person_likes,
                Dislikes=person_dislikes,
                Hobbies=person_hobbies,
                Mother_FName=mother_FName,
                Mother_LName=family_LName,
                Father_FName=father_FName,
                Father_LName=family_LName
            ),
            "Summary": content_dict[key]
        }

    data_dict = {
        "Name": content_dict["person"],
        "Content": content,
        "Context": context_data
    }

    return data_dict

def main():
    person_likes_list = load_data(csv_fpath="./Likes.csv")
    person_likes_provider = DynamicProvider(
        provider_name="person_likes",
        elements=person_likes_list)

    person_dislikes_list = load_data(csv_fpath="./Dislikes.csv")
    person_dislikes_provider = DynamicProvider(
        provider_name="person_dislikes",
        elements=person_dislikes_list)

    person_hobbies_list = load_data(csv_fpath="./Hobbies.csv")
    person_hobbies_provider = DynamicProvider(
        provider_name="person_hobbies",
        elements=person_hobbies_list)

    kenya_locations_list = load_data(csv_fpath="./Locations.csv")
    kenya_locations_provider = DynamicProvider(
        provider_name="kenya_locations",
        elements=kenya_locations_list)

    data_generator = Faker()

    data_generator.add_provider(person_likes_provider)
    data_generator.add_provider(person_dislikes_provider)
    data_generator.add_provider(kenya_locations_provider)
    data_generator.add_provider(person_hobbies_provider)

    json_fpath = "./Dataset_template.json"
    with open(json_fpath) as json_f:
        data_json = json.load(json_f)

    data_dict = generate_data(
        data_generator=data_generator,
        data_json=data_json)

if __name__ == "__main__":
    main()
