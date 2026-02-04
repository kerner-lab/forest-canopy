"""
Utility functions to use across models for finetuning
"""

def channel_adapter(expected_channels, img_channels):
    """
    Tool to adapt channels from expected for backbone model to
    the number of channels provided by the dataset in finetuning.
    learnable 1×1 convolution that transforms input from one number of
    channels to another
    :return:
    """
    if img_channels != expected_channels:
        input_adapter = nn.Conv2d(
            img_channels,
            expected_channels,
            kernel_size=1,
            bias=False,
        )
    else:
        input_adapter = nn.Identity()

    return input_adapter
