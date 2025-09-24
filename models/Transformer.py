import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerBlock,
    PositionalEncoding)


"""
(Decoder-only / Encoder-Decoder) Transformer model.
(Model_0, Model_1, or Model_2) Architecture.
"""
class Transformer(nn.Module):
    def __init__(
            self,
            num_decoder_embeddings,
            embedding_dim,
            hidden_dim,
            num_heads=8,
            num_encoder_embeddings=None,
            num_encoder_blocks=3,
            num_decoder_blocks=6,
            out_classes=8,
            use_cross_attn=False,
            activation_type="gelu"):
        super().__init__()

        self.use_cross_attn = use_cross_attn

        # Learnable Embedding and Positional Encoding.
        if self.use_cross_attn:
            self.encoder_emb_layer = nn.Embedding(
                num_embeddings=num_encoder_embeddings,
                embedding_dim=embedding_dim)
        self.decoder_emb_layer = nn.Embedding(
            num_embeddings=num_decoder_embeddings,
            embedding_dim=embedding_dim)
        self.pos_layer = PositionalEncoding()

        if self.use_cross_attn:
            # Encoder Model Blocks.
            self.encoder_model_blocks = nn.ModuleList()
            for _ in range(num_encoder_blocks):
                self.encoder_model_blocks.append(
                    TransformerBlock(
                        heads=num_heads,
                        hidden_dim=hidden_dim,
                        embedding_dim=embedding_dim,
                        is_causal=False,
                        use_cross_attn=False,
                        activation_type=activation_type))

        # Decoder Model Blocks.
        self.decoder_model_blocks = nn.ModuleList()
        for _ in range(num_decoder_blocks):
            self.decoder_model_blocks.append(
                TransformerBlock(
                    heads=num_heads,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    is_causal=True,
                    use_cross_attn=use_cross_attn,
                    activation_type=activation_type))

        # Classifier Block.
        self.classifier_block = nn.Sequential(
            LinearBlock(
                in_dim=embedding_dim,
                out_dim=hidden_dim,
                use_activation=True),
            LinearBlock(
                in_dim=hidden_dim,
                out_dim=out_classes,
                use_activation=False))

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # Backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward_embeddings(self, x, y):
        # Embedding Layer + Positional Encoding.
        y_pos = None
        if self.use_cross_attn:
            y_emb = self.encoder_emb_layer(y)
            y_pos = self.pos_layer(y_emb)

        x_emb = self.decoder_emb_layer(x)
        x_pos = self.pos_layer(x_emb)

        return x_pos, y_pos

    def forward_model(self, x, y):
        # Encoder Model Blocks.
        if self.use_cross_attn:
            for encoder_model_block in self.encoder_model_blocks:
                y = encoder_model_block(y)

        # Decoder Model Blocks.
        for decoder_model_block in self.decoder_model_blocks:
            x = decoder_model_block(x, y)

        return x

    def forward_classifier(self, x):
        x_classifier = self.classifier_block(x)

        return x_classifier

    def forward(self, x, y=None):
        x_emb_pos, y_emb_pos = self.forward_embeddings(x, y)
        x_model = self.forward_model(x_emb_pos, y_emb_pos)
        x_classifier = self.forward_classifier(x_model)

        return x_classifier
