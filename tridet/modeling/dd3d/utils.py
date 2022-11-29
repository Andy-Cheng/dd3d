
def get_fpn_out_channels(output_shape):
    out_channels = []
    if isinstance(output_shape, list):
        out_channels = [x.channels for x in output_shape]
    elif isinstance(output_shape, dict):
        out_channels = [x.channels for x in output_shape.values()]
    assert len(set(out_channels)) == 1, "The feature extractor must produce same channels of features for all levels."
    out_channels = out_channels[0]
    return out_channels
