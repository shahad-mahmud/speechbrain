"""A custom CRDNN model for streaming ASR. The motive behind this model is to
return the hidden state of RNN layer(s) during the forward pass. Original CRDNN 
model can be found in: `speechbrain/lobes/models/CRDNN.py`

Author:
 * Md. Shahad Mahmud Chowdhury 2021
"""
import torch
import speechbrain as sb


class CustomCRDNN(torch.nn.Module):
    def __init__(
        self,
        input_size=None,
        input_shape=None,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        cnn_blocks=2,
        cnn_channels=[128, 256],
        cnn_kernelsize=(3, 3),
        cnn_padding='same',  # or 'valid
        inter_layer_pooling_size=[2, 2],
        using_2d_pooling=False,
    ) -> None:
        if input_size is None and input_shape is None:
            raise ValueError("Must specify one of input_size or input_shape")

        if input_shape is None:
            input_shape = [None, None, input_size]
        super().__init__()
        self.input_size = input_shape[-1]

        self.CNN = None
        if cnn_blocks > 0:
            self.CNN = sb.nnet.containers.Sequential(input_shape=input_shape)
            for block_index in range(cnn_blocks):
                self.CNN.append(
                    Conv2dUnit,
                    channels=cnn_channels[block_index],
                    kernel_size=cnn_kernelsize,
                    using_2d_pool=using_2d_pooling,
                    pooling_size=inter_layer_pooling_size[block_index],
                    activation=activation,
                    dropout=dropout,
                    padding=cnn_padding,
                    layer_name=f"block_{block_index}",
                )


class Conv2dUnit(sb.nnet.containers.Sequential):
    def __init__(
        self,
        input_shape,
        channels,
        kernel_size=[3, 3],
        activation=torch.nn.LeakyReLU,
        using_2d_pool=False,
        pooling_size=2,
        dropout=0.15,
        padding='valid'
    ):
        super().__init__(input_shape=input_shape)
        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=channels,
            kernel_size=kernel_size,
            layer_name="conv_1",
            padding=padding,
        )
        self.append(sb.nnet.normalization.LayerNorm, layer_name="norm_1")
        self.append(activation(), layer_name="act_1")
        self.append(
            sb.nnet.CNN.Conv2d,
            out_channels=channels,
            kernel_size=kernel_size,
            layer_name="conv_2",
            padding=padding,
        )
        self.append(sb.nnet.normalization.LayerNorm, layer_name="norm_2")
        self.append(activation(), layer_name="act_2")

        if using_2d_pool:
            self.append(
                sb.nnet.pooling.Pooling2d(
                    pool_type="max",
                    kernel_size=(pooling_size, pooling_size),
                    pool_axis=(1, 2),
                ),
                layer_name="pooling",
            )
        else:
            self.append(
                sb.nnet.pooling.Pooling1d(
                    pool_type="max",
                    input_dims=4,
                    kernel_size=pooling_size,
                    pool_axis=2,
                ),
                layer_name="pooling",
            )

        self.append(
            sb.nnet.dropout.Dropout2d(drop_rate=dropout), layer_name="drop"
        )
