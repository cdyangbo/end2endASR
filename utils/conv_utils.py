#-*- coding:utf-8 -*-
'''
some function from keras
'''



def conv_output_length(input_length, filter_size,padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.

    # Returns
        The output length (integer).
    """

    padding = padding.lower()
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
    """Determines input length of a convolution given output length.

    # Arguments
        output_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.

    # Returns
        The input length (integer).
    """
    padding = padding.lower()
    if output_length is None:
        return None
    assert padding in {'same', 'valid', 'full'}
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    elif padding == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size


def deconv_length(dim_size, stride_size, kernel_size, padding):
    if dim_size is None:
        return None
    padding = padding.lower()
    if padding == 'valid':
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == 'full':
        dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
    elif padding == 'same':
        dim_size = dim_size * stride_size
    return dim_size
