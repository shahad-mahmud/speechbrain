"""A custom CRDNN model for streaming ASR. The motive behind this model is to
return the hidden state of RNN layer(s) during the forward pass. Original CRDNN 
model can be found in: `speechbrain/lobes/models/CRDNN.py`

Author:
 * Md. Shahad Mahmud Chowdhury 2021
"""
import torch
import speechbrain as sb
from speechbrain.lobes.models.CRDNN import DNN_Block


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
        time_pooling=False,
        time_pooling_size=2,
        rnn_class=sb.nnet.RNN.LiGRU,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_re_init=False,
        dnn_blocks=2,
        dnn_neurons=512,
    ) -> None:
        if input_size is None and input_shape is None:
            raise ValueError("Must specify one of input_size or input_shape")

        if input_shape is None:
            input_shape = [None, None, input_size]
        super().__init__()
        self.input_shape = input_shape

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

        self.time_pooling = None
        if time_pooling:
            self.time_pooling = sb.nnet.pooling.Pooling1d(
                pool_type="max",
                input_dims=4,
                kernel_size=time_pooling_size,
                pool_axis=1,
            )

        self.rnn = None
        if rnn_layers > 0:
            self.rnn = rnn_class(
                hidden_size=rnn_neurons,
                input_shape=self.get_rnn_input_shape(),
                num_layers=rnn_layers,
                dropout=dropout,
                bidirectional=False,
                re_init=rnn_re_init,
            )

        self.dnn = None
        if dnn_blocks > 0:
            self.dnn = sb.nnet.containers.Sequential(
                input_shape=[None, None, rnn_neurons])
            for block_index in range(dnn_blocks):
                self.dnn.append(
                    DNN_Block,
                    neurons=dnn_neurons,
                    activation=activation,
                    dropout=dropout,
                    layer_name=f"block_{block_index}",
                )

    def forward(self, input, rnn_hidden=None):
        output = input
        if self.CNN is not None:
            output = self.CNN(output)
        if self.time_pooling is not None:
            output = self.time_pooling(output)
        if self.rnn is not None:
            if len(output.shape) == 4:
                output = output.reshape(
                    output.shape[0], output.shape[1], output.shape[2] * output.shape[3])
            output, rnn_hidden = self.rnn(
                output) if rnn_hidden is None else self.rnn(output, rnn_hidden)
        if self.dnn is not None:
            output = self.dnn(output)

        return output, rnn_hidden

    def get_rnn_input_shape(self):
        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            if self.CNN is not None:
                output = self.CNN(dummy_input)
            if self.time_pooling is not None:
                output = self.time_pooling(output)
        return output.shape


class Conv2dUnit(sb.nnet.containers.Sequential):
    """2D convolutional layer based on CNN_Block form the CRDNN model. In this
    class an option to pass padding type is passed. This is useful as we need 
    the shape to be same for all summed audio chunks and the whole audio.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    channels : int
        Number of convolutional channels for the block.
    kernel_size : tuple
        Size of the 2d convolutional kernel
    activation : torch.nn.Module class
        A class to be used for instantiating an activation layer.
    using_2d_pool : bool
        Whether to use 2d pooling or only 1d pooling.
    pooling_size : int
        Size of pooling kernel, duplicated for 2d pooling.
    dropout : float
        Rate to use for dropping channels.
    padding : str
        `same` or `valid` (default). If `same`, the output shape will be the
        same as input shape. If `valid`, no padding is added.
        
    TODO: Add example
        
    """
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
