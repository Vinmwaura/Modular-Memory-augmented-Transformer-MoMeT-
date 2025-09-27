import torch
import torch.nn.functional as F

from models.Transformer import Transformer

from dataset_loader.token_dataset import TokenDataset

def generate_text(
        device,
        model,
        context_window,
        model_type,
        special_tokens,
        end_special_tokens,
        input_data,
        inverted_vocabulary,
        encoder_data=None,
        temperature=0.75):
    # Convert Encoder tokens to tensor.
    encoder_data_tensor = None
    if encoder_data is not None:
        encoder_data_tensor = torch.tensor(
            [encoder_data],
            device=device,
            dtype=torch.long)

    model.eval()

    gen_data = []
    last_token_predicted = None
    while len(input_data) < context_window:
        # Convert Decoder tokens to tensor.
        input_data_tensor = torch.tensor(
            [input_data],
            device=device,
            dtype=torch.long)

        with torch.no_grad():
            out_classifier = model(
                x=input_data_tensor,
                y=encoder_data_tensor)

        # Probabilities.
        probs = F.softmax(out_classifier[0] / temperature, dim=1)

        # Pick most likely token for next generation for each Token Sequence (Seq).
        next_token = torch.multinomial(probs, 1).squeeze(1)

        # Save last token for next prediction.
        last_token_predicted = next_token[-1].item()
        input_data.append(last_token_predicted)

        if last_token_predicted == end_special_tokens:
            break

        gen_data.append(last_token_predicted)

    clean_pred_token = [token for token in gen_data if token not in special_tokens]
    pred_token_list = [inverted_vocabulary[token_id] for token_id in clean_pred_token]
    predicted_text = "".join(pred_token_list)
    
    return predicted_text
