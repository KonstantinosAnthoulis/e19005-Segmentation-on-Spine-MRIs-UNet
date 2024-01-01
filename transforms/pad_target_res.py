import torch

def pad_to_resolution(input_tensor, target_resolution):
    current_resolution = input_tensor.shape[-2:]

    if current_resolution == target_resolution:
        return input_tensor  # No padding needed

    pad_height = target_resolution[0] - current_resolution[0]
    pad_width = target_resolution[1] - current_resolution[1]

    # Calculate pad values
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Apply padding
    padded_tensor = torch.nn.functional.pad(input_tensor, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)

    return padded_tensor