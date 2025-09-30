import os
import csv
import json
import random
import pathlib
import argparse

import torch
import torch.nn.functional as F

from models.Transformer import Transformer

from utils.model_utils import load_model
from utils.generation_utils import generate_text

# Loads list of items to be randomly selected for the template.
def load_data(csv_fpath, delimiter='\n'):
    with open(csv_fpath) as csv_f:
        reader = csv.reader(csv_f, delimiter=delimiter)
        data_list = list(reader)

    data_list_flat = list(chain.from_iterable(data_list))
    return data_list_flat

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.001:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    project_name = "Proof-Of-Concept: Transformer model(s) text generation"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--vocabulary-path",
        help="File path to vocabulary.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-0-checkpoint",
        help="File path to model_0 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-1-checkpoint",
        help="File path to model_1 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--model-2-checkpoint",
        help="File path to model_2 checkpoint.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--tst-dataset-path",
        help="File path to testing json dataset file.",
        required=True,
        type=pathlib.Path)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    temperature = args["temperature"]  # Temperature value.
    vocabulary_path = args["vocabulary_path"]  # Vocabulary json file path (*.json).
    tst_dataset_path = args["tst_dataset_path"]  # Training json file path (*.json).
    model_0_checkpoint = args["model_0_checkpoint"]
    model_1_checkpoint = args["model_1_checkpoint"]
    model_2_checkpoint = args["model_2_checkpoint"]

    # Load Vocabulary dataset.
    with open(vocabulary_path, "r") as json_f:
        vocabulary_dict = json.load(json_f)

    # Inverted vocabulary: id_to_tokens.
    inverted_vocabulary = {}
    for token, token_id in vocabulary_dict["tokens_to_id"].items():
        inverted_vocabulary[token_id] = token

    # Special Tokens.
    special_tokens = vocabulary_dict["special_tokens_to_id"]
    special_tokens_list = list(special_tokens.values())

    # Testing Datasets.
    with open(tst_dataset_path, "r") as json_f:
        json_data = json.load(json_f)

    # Testing Dataset categories and filepaths.
    tst_fpaths_list = json_data["fpaths"]
    tst_categories_list = json_data["categories"]

    # Generate text using randomly selected testing dataset and model.
    random_tst_fpath = random.choice(tst_fpaths_list)
    with open(random_tst_fpath, "r") as json_f:
        tst_json_data = json.load(json_f)

    # Context Tokens.
    context_dict = tst_json_data["context"]

    # Randomly pick a category for the prompt.
    random_category = random.choice(tst_categories_list)

    # Content Tokens.
    content_tokens = tst_json_data["content"]
    content_token_list = [inverted_vocabulary[token_id] for token_id in content_tokens]
    content_text = "".join(content_token_list)

    # Prompt.
    prompt_tokens = context_dict[random_category]["prompt"]
    prompt_token_list = [inverted_vocabulary[token_id] for token_id in prompt_tokens]
    prompt_text = "".join(prompt_token_list)

    print("=" * 100)
    print(f"Content => {content_text}")
    print(f"Prompt => {prompt_text}")
    print("*" * 100)

    models_fpaths_list = [
        model_0_checkpoint,
        model_1_checkpoint,
        model_2_checkpoint]

    for model_type, model_checkpoint_fpath in enumerate(models_fpaths_list):
        print(f"Loading pre-trained Model_{model_type}.")
        print("-" * 30)

        classifier_status, classifier_dict = load_model(model_checkpoint_fpath)
        if not classifier_status:
            raise Exception(f"An error occured while loading pretrained Model_{model_type} checkpoint!")

        num_decoder_embeddings = classifier_dict["num_decoder_embeddings"]
        num_encoder_embeddings = classifier_dict["num_encoder_embeddings"]
        embedding_dim = classifier_dict["embedding_dim"]
        hidden_dim = classifier_dict["hidden_dim"]
        num_heads = classifier_dict["num_heads"]
        num_encoder_blocks = classifier_dict["num_encoder_blocks"]
        num_decoder_blocks = classifier_dict["num_decoder_blocks"]
        out_classes = classifier_dict["num_decoder_embeddings"]
        # use_cross_attn = classifier_dict["use_cross_attn"]
        activation_type = classifier_dict["activation_type"]

        use_cross_attn = False
        if model_type != 0:
            use_cross_attn = True

        # Transformer model.
        model = Transformer(
            special_tokens=special_tokens,
            num_decoder_embeddings=num_decoder_embeddings,
            num_encoder_embeddings=num_encoder_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            out_classes=num_decoder_embeddings,
            use_cross_attn=use_cross_attn,
            activation_type=activation_type)

        model.custom_load_state_dict(classifier_dict["model"])
        model = model.to(device)

        context_window = classifier_dict["context_window"]

        # Prepend and append special tokens to the prompt token.
        input_prompt_tokens = [special_tokens["start_prompt"]] + prompt_tokens + [special_tokens["end_prompt"]]
        if model_type == 0:
            end_special_tokens = special_tokens["end_tag"]
            input_prompt_tokens = input_prompt_tokens + [special_tokens["start_tag"]]
            encoder_prompt_tokens = None
        elif model_type == 1:
            end_special_tokens = special_tokens["EContext"]
            input_prompt_tokens = input_prompt_tokens + [special_tokens["SContext"]]
            encoder_prompt_tokens = tst_json_data["content"]
        elif model_type == 2:
            end_special_tokens = special_tokens["end_response"]
            input_prompt_tokens = input_prompt_tokens + [special_tokens["start_response"]]
            encoder_prompt_tokens = tst_json_data["context"][random_category]["summary"]

        # Conditional text passed as conditional input to the Encoder model.
        if encoder_prompt_tokens != None:
            encoder_prompt_token_list = [inverted_vocabulary[token_id] for token_id in encoder_prompt_tokens]
            encoder_prompt_txt = "".join(encoder_prompt_token_list)
            print(f"Conditional Input => {encoder_prompt_txt}")

        # Response from the model.
        model_response = generate_text(
            device=device,
            model=model,
            context_window=context_window,
            model_type=model_type,
            special_tokens=special_tokens_list,
            end_special_tokens=end_special_tokens,
            input_data=input_prompt_tokens,
            encoder_data=encoder_prompt_tokens,
            inverted_vocabulary=inverted_vocabulary,
            temperature=temperature)
        
        if model_type == 0:
            print(f"Model Output (Named-Entity Recognition) => \"{model_response}\"")
        elif model_type == 1:
            print(f"Model Output (Summarization) => \"{model_response}\"")
        elif model_type == 2:
            print(f"Model Output (Text Generation) => \"{model_response}\"")
        print("*" * 100)

if __name__ == "__main__":
    main()
