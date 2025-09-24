import json
import random

import torch
from torch.utils.data import Dataset

"""
Load Token Dataset for Model_0, Model_1, Model_2.
"""
class TokenDataset(Dataset):
    def __init__(self, json_dataset, context_window, special_tokens):
        self.special_tokens = special_tokens
        self.context_window = context_window

        with open(json_dataset, "r") as json_f:
            dataset = json.load(json_f)

        self.categories = dataset["categories"]
        self.fpaths_list = dataset["fpaths"]

    def __len__(self):
        return len(self.fpaths_list)

    def __getitem__(self, index):
        json_fpath = self.fpaths_list[index]

        with open(json_fpath, "r") as json_f:
            json_data = json.load(json_f)

        tag_tokens = json_data["tag"]
        content_tokens = json_data["content"]
        context_dict = json_data["context"]

        # Randomly pick a category to form the basis for Prompts, Responses, Summaries.
        random_category = random.choice(self.categories)

        # Input and Target tokens for Model_0.
        model_0_tokens = [self.special_tokens["start_prompt"]] + \
            context_dict[random_category]["prompt"] + \
            [self.special_tokens["end_prompt"]] + \
            [self.special_tokens["start_tag"]] + \
            tag_tokens + \
            [self.special_tokens["end_tag"]]

        # Pad token list to be of unform length.
        model_0_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_0_tokens) + 1)
        in_model_0_tokens = model_0_tokens[:-1] + model_0_paddings
        target_model_0_tokens = model_0_tokens[1:] + model_0_paddings

        # Input and Target tokens for Model_1.        
        model_1_tokens = [self.special_tokens["start_prompt"]] + \
            context_dict[random_category]["prompt"] + \
            [self.special_tokens["end_prompt"]] + \
            [self.special_tokens["SContext"]] + \
            context_dict[random_category]["summary"] + \
            [self.special_tokens["EContext"]]

        # Pad token list to be of unform length.
        model_1_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_1_tokens) + 1)
        in_model_1_tokens = model_1_tokens[:-1] + model_1_paddings
        target_model_1_tokens = model_1_tokens[1:] + model_1_paddings

        # Input and Target tokens for Model_2.        
        model_2_tokens = [self.special_tokens["start_prompt"]] + \
            context_dict[random_category]["prompt"] + \
            [self.special_tokens["end_prompt"]] + \
            [self.special_tokens["start_response"]] + \
            context_dict[random_category]["response"] + \
            [self.special_tokens["end_response"]]

        # Pad token list to be of unform length.
        model_2_paddings = [self.special_tokens["pad_token"]] * (self.context_window - len(model_2_tokens) + 1)
        in_model_2_tokens = model_2_tokens[:-1] + model_2_paddings
        target_model_2_tokens = model_2_tokens[1:] + model_2_paddings

        tensor_dict = {
            "model_0": {
                "in": torch.tensor(in_model_0_tokens).long(),
                "target": torch.tensor(target_model_0_tokens).long(),
            },
            "model_1": {
                "in": torch.tensor(in_model_1_tokens).long(),
                "target": torch.tensor(target_model_1_tokens).long(),
            },
            "model_2": {
                "in": torch.tensor(in_model_2_tokens).long(),
                "target": torch.tensor(target_model_2_tokens).long(),
            },
        }
        return tensor_dict
