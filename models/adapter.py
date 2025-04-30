import torch.nn as nn


class Efficient_LSTM_Pass(nn.Module):
    def __init__(self) -> None:
        super(Efficient_LSTM_Pass, self).__init__()
        self.adapter_down = nn.Linear(768, 32)  # equivalent to 1 * 1 Conv
        self.adapter_gelu = nn.GELU()
        self.adapter_lstm = nn.LSTM(input_size=32, hidden_size=32, batch_first=True, num_layers=1)
        self.adapter_up = nn.Linear(32, 768)  # equivalent to 1 * 1 Conv

        self.adapter_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        down = self.adapter_gelu(self.adapter_down(x))  # shape = [batch_size, 64, 128]
        # down = down.permute(0, 2, 1)  # shape = [batch_size, 128, 64]
        conv = self.adapter_gelu(self.adapter_lstm(down)[0])
        # conv = conv.permute(0, 2, 1)  # shape = [batch_size, 64, 128]
        up = self.adapter_gelu(self.adapter_up(conv))  # shape = [batch_size, 64, 768]

        out = self.adapter_norm(up + x)  # shape = [batch_size, 64, 768]
        return out


class Efficient_Conv_Pass(nn.Module):
    def __init__(
        self,
        hidden_dim,
        adapter_hidden_dim,
    ):
        super(Efficient_Conv_Pass, self).__init__()

        # More efficient 1d conv - 492k params per encoder layer
        self.adapter_down = nn.Linear(hidden_dim, adapter_hidden_dim)  # equivalent to 1 * 1 Conv
        self.adapter_gelu = nn.GELU()
        self.adapter_1d_cnn = nn.Conv1d(
            in_channels=adapter_hidden_dim,
            out_channels=adapter_hidden_dim,
            kernel_size=3,
            stride=1,
            padding="same",
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.adapter_up = nn.Linear(adapter_hidden_dim, hidden_dim)  # equivalent to 1 * 1 Conv

        self.adapter_norm = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        down = self.adapter_gelu(self.adapter_down(x))  # shape = [batch_size, 64, 128]
        down = down.permute(0, 2, 1)  # shape = [batch_size, 128, 64]
        conv = self.adapter_gelu(self.adapter_1d_cnn(down))  # shape = [batch_size, 128, 64]
        conv = conv.permute(0, 2, 1)  # shape = [batch_size, 64, 128]
        up = self.adapter_gelu(self.adapter_up(conv))  # shape = [batch_size, 64, 768]

        out = self.adapter_norm(up + x)  # shape = [batch_size, 64, 768]
        return out


class Efficient_MLP_Pass(nn.Module):
    def __init__(self) -> None:
        super(Efficient_MLP_Pass, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )
        # self.model = nn.Sequential(
        #     nn.Dropout(p=0.1, inplace=False),
        #     nn.Linear(in_features=64, out_features=16, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=64, bias=True),
        # )
        # self.ln = nn.Sequential(
        #     nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        # )

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.model(x)
        # x = x.permute(0, 2, 1)
        # x = self.ln(x)
        return x


################# Non-Linear Adapters ######################

# The adapter is placed in-between MHSA and FFN with skip connection, instead of a parallel configuration


class w2v2_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_nlp, self).__init__()

        self.attention = transformer_encoder.attention
        self.drop = transformer_encoder.dropout
        self.ln1 = transformer_encoder.layer_norm

        self.adapter = Efficient_MLP_Pass()
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):
        attention, _ = self.attention(x)
        attention = self.drop(attention)
        attention = self.ln1(attention)
        mhsa = attention + x
        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


class vit_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_nlp, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        self.adapter = Efficient_MLP_Pass()

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        mhsa = self.drop(attention) + x

        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


class w2v2_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder, hidden_dim, adapter_hidden_dim):
        super(w2v2_adapter_conv, self).__init__()

        # Attention Layers
        # note that for W2V2 encoder layer_norm is after MHSA while for ViT encoder layer_norm is before
        self.attention = transformer_encoder.attention
        self.drop = transformer_encoder.dropout
        self.ln1 = transformer_encoder.layer_norm

        self.mhsa_conv_pass = Efficient_Conv_Pass(hidden_dim, adapter_hidden_dim)
        self.ffn_conv_pass = Efficient_Conv_Pass(hidden_dim, adapter_hidden_dim)

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, hidden_dim]
        attention, _ = self.attention(x)
        attention = self.drop(attention)
        attention = self.ln1(attention)

        mhsa = attention + x + self.mhsa_conv_pass(x)
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class vit_adapter_lstm(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_lstm, self).__init__()

        # Attention Layers. refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_lstm_pass = Efficient_LSTM_Pass()
        self.ffn_lstm_pass = Efficient_LSTM_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, 768]

        conv_pass = self.mhsa_lstm_pass(x)

        # Attention need to be performed individually. Refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_lstm_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class w2v2_adapter_lstm(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_lstm, self).__init__()

        # Attention Layers
        # note that for W2V2 encoder layer_norm is after MHSA while for ViT encoder layer_norm is before
        self.attention = transformer_encoder.attention
        self.drop = transformer_encoder.dropout
        self.ln1 = transformer_encoder.layer_norm

        self.mhsa_lstm_pass = Efficient_LSTM_Pass()
        self.ffn_lstm_pass = Efficient_LSTM_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, 768]
        attention, _ = self.attention(x)
        attention = self.drop(attention)
        attention = self.ln1(attention)

        mhsa = attention + x + self.mhsa_lstm_pass(x)
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_lstm_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class vit_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_conv, self).__init__()

        # Attention Layers. refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = Efficient_Conv_Pass(768, 32)
        self.ffn_conv_pass = Efficient_Conv_Pass(768, 32)

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, 768]

        conv_pass = self.mhsa_conv_pass(x)

        # Attention need to be performed individually. Refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class video_mae_adapter_layer_conv(nn.Module):
    def __init__(self, transformer_encoder_layer, hidden_dim, adapter_hidden_dim) -> None:
        super(video_mae_adapter_layer_conv, self).__init__()
        self.transformer_encoder_layer = transformer_encoder_layer

        self.mhsa_conv_pass = Efficient_Conv_Pass(hidden_dim, adapter_hidden_dim)
        self.ffn_conv_pass = Efficient_Conv_Pass(hidden_dim, adapter_hidden_dim)

        self.adapter_norm1 = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)

    def forward(self, hidden_states):
        conv_pass = self.mhsa_conv_pass(hidden_states)

        self_attention_outputs = self.transformer_encoder_layer.attention(
            self.transformer_encoder_layer.layernorm_before(
                hidden_states
            ),  # in VideoMAE, layernorm is applied before self-attention
            None,
            output_attentions=False,
        )
        attention_output = self_attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states + conv_pass
        hidden_states = self.adapter_norm1(hidden_states)

        # in VideoMAE, layernorm is also applied after self-attention
        layer_output = self.transformer_encoder_layer.layernorm_after(hidden_states)
        layer_output = self.transformer_encoder_layer.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.transformer_encoder_layer.output(layer_output, hidden_states)

        outputs = layer_output + self.ffn_conv_pass(hidden_states) + hidden_states
        # outputs = layer_output + conv_pass + hidden_states
        outputs = self.adapter_norm2(outputs)

        return outputs


class video_mae_adapter_layer_conv_multi(nn.Module):
    def __init__(self, transformer_encoder_layers, hidden_dim, adapter_hidden_dim) -> None:
        super(video_mae_adapter_layer_conv_multi, self).__init__()
        self.transformer_encoder_layers = transformer_encoder_layers

        self.conv_pass = Efficient_Conv_Pass(hidden_dim, adapter_hidden_dim)

        self.adapter_norm = nn.LayerNorm((hidden_dim,), eps=1e-05, elementwise_affine=True)

    def forward(self, hidden_states):
        conv_pass = self.conv_pass(hidden_states.clone())
        layer_output = hidden_states.clone()
        for layer in self.transformer_encoder_layers:
            layer_output = layer(layer_output)[0]

        outputs = conv_pass + layer_output + hidden_states
        # outputs = layer_output + conv_pass + hidden_states
        outputs = self.adapter_norm(outputs)

        return outputs
