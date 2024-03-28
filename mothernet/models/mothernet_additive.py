import torch
import torch.nn as nn

from mothernet.models.decoders import AdditiveModelDecoder, FactorizedAdditiveModelDecoder
from mothernet.models.encoders import BinEmbeddingEncoder, Linear, OneHotAndLinear
from mothernet.models.layer import TransformerEncoderLayer
from mothernet.models.tabpfn import TransformerEncoderDiffInit
from mothernet.utils import SeqBN, get_init_method


class MotherNetAdditive(nn.Module):
    def __init__(self, *, n_features, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_embed_dim=2048, low_rank_weights=None, weight_embedding_rank=None,
                 decoder_hidden_layers=1, decoder_hidden_size=None, n_bins=64, input_bin_embedding=False,
                 bin_embedding_rank=16, output_rank=16, factorized_output=False, y_encoder=None,
                 predicted_hidden_layer_size=None, predicted_hidden_layers=None,
                 decoder_type=None, input_layer_norm=False, shape_attention=False, tabpfn_zero_weights=True, shape_attention_heads=1, n_shape_functions=32,
                 shape_init="constant", decoder_activation='relu', fourier_features=0):
        super().__init__()
        nhid = emsize * nhid_factor
        self.y_encoder = y_encoder_layer
        self.low_rank_weights = low_rank_weights  # ignored for now
        self.weight_embedding_rank = weight_embedding_rank  # ignored for now
        self.decoder_activation = decoder_activation

        assert fourier_features == 0, "Fourier features are not supported in this model yet"

        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer_creator(), nlayers)\
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.emsize = emsize

        if input_bin_embedding == "linear":
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=False,
                                               decoder_activation=decoder_activation)
        elif input_bin_embedding in ["True", "nonlinear"] or isinstance(input_bin_embedding, bool) and input_bin_embedding:
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=True,
                                               decoder_activation=decoder_activation)
        elif input_bin_embedding in ["none", "False"] or isinstance(input_bin_embedding, bool) and not input_bin_embedding:
            self.encoder = nn.Sequential(nn.Flatten(-2, -1), Linear(num_features=n_features*n_bins, emsize=emsize, replace_nan_by_zero=True))
        else:
            raise ValueError(f"Unknown input_bin_embedding: {input_bin_embedding}")

        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_bins = n_bins
        self.n_out = n_out
        self.nhid = nhid
        self.input_bin_embedding = input_bin_embedding
        self.output_rank = output_rank
        self.factorized_output = factorized_output
        self.decoder_type = decoder_type
        self.input_layer_norm = input_layer_norm
        self.shape_attention = shape_attention
        self.tabpfn_zero_weights = tabpfn_zero_weights

        if factorized_output:
            self.decoder = FactorizedAdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                          embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                          decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, rank=output_rank,
                                                          shape_attention=shape_attention, shape_attention_heads=shape_attention_heads,
                                                          n_shape_functions=n_shape_functions, shape_init=shape_init, decoder_activation=decoder_activation,)
        else:
            self.decoder = AdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, decoder_activation=decoder_activation,)

        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))
        if self.input_layer_norm:
            self.input_norm = nn.LayerNorm(normalized_shape=(n_features, n_bins))
        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        _, x_src_org, y_src_org = src
        X_onehot, _ = bin_data(x_src_org, n_bins=self.n_bins,
                               single_eval_pos=single_eval_pos)
        X_onehot = X_onehot.float()
        if self.input_layer_norm:
            X_onehot_norm = self.input_norm(X_onehot)
        else:
            X_onehot_norm = X_onehot
        x_src = self.encoder(X_onehot_norm)
        y_src = self.y_encoder(y_src_org.unsqueeze(-1) if len(y_src_org.shape) < len(x_src.shape) else y_src_org)
        enc_train = x_src[:single_eval_pos] + y_src[:single_eval_pos]

        # FIXME Refactor into a function to share with mothernet
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat([self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0)
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError("class_tokens decoder type is only supported with OneHotAndLinear y_encoder")
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        output = self.transformer_encoder(enc_train)
        weights, biases = self.decoder(output, y_src_org[:single_eval_pos])
        # n samples, b batch, k feature, d bins, o outputs
        h = torch.einsum("nbkd,bkdo->nbo", X_onehot[single_eval_pos:], weights)
        h = h + biases

        if h.isnan().all():
            print("NAN")
            import pdb
            pdb.set_trace()
        return h


def bin_data(data, n_bins, single_eval_pos=None):
    # data is samples x batch x features
    quantiles = torch.arange(n_bins - 1, device=data.device) / (n_bins - 2)

    # Compute quantiles without nan data
    if single_eval_pos is None:
        bin_edges = torch.nanquantile(data, quantiles[1:-1], dim=0)
    else:
        bin_edges = torch.nanquantile(data[:single_eval_pos], quantiles[1:-1], dim=0)

    # FIXME extra data copy
    bin_edges = bin_edges.transpose(0, -1).contiguous()
    data = data.transpose(0, -1).contiguous()

    # Keep track of the nan positions in the data
    isnan = torch.isnan(data)

    # Fill NaNs in order to bin the data.
    data = torch.nan_to_num(data, nan=0.0)
    X_binned = torch.searchsorted(bin_edges, data)

    # Put NaN data on the last bin.
    X_binned[isnan] = n_bins - 1
    X_onehot = nn.functional.one_hot(X_binned.transpose(0, -1), num_classes=n_bins)
    return X_onehot, bin_edges
