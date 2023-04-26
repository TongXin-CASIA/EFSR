import torch
from torch.nn import functional as F


def generate_absolute_optical_flow(relative_flow):
    """
    Generate absolute optical flow from relative optical flow.
    Args:
        relative_flow: relative optical flow
    Returns:
        absolute optical flow [B, H, W, [x, y]]
    """
    w, h = relative_flow.size()[-1], relative_flow.size()[-2]
    relative_flow = relative_flow.permute(0, 2, 3, 1)
    relative_flow[:, :, :, 0] = relative_flow[:, :, :, 0] * 2.0 / (w - 1)
    relative_flow[:, :, :, 1] = relative_flow[:, :, :, 1] * 2.0 / (h - 1)
    base_flow = F.affine_grid(torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=relative_flow.device), (1, 1, h, w))
    return relative_flow + base_flow


def warp_image(image, flow, mode="bilinear"):
    """
    Warp image with optical flow.
    Args:
        image: image to warp
        flow: optical flow
        mode: interpolate mode
    Returns:
        warped image
    """
    return F.grid_sample(image, flow, mode=mode)


def warp_image_relative(image, flow, mode="bilinear"):
    """
    Warp image with relative optical flow.
    Args:
        image: image to warp
        flow: relative optical flow
    Returns:
        warped image
    """
    return warp_image(image, generate_absolute_optical_flow(flow), mode)


def generate_relative_optical_flow_multi_octave(image, flow_list):
    """
    Generate absolute optical flow from relative optical flow.
    Args:
        relative_flow_list: list of relative optical flow
    Returns:
        absolute optical flow
    """
    size = image.size()
    # resize flow to image size
    flow_list_resized = [F.interpolate(flow, (size[2], size[3]), mode='bilinear') for flow in flow_list]
    flow = flow_list_resized[-1]
    for i in range(1, len(flow_list)):
        flow += flow_list_resized[-1 - i]
    return flow


def warp_image_two_steps(image, flow_first, flow_second, mode='bilinear'):
    """
    the function equal to warp_image(warp_image(image1, flow_first), flow_second)
    :param image:
    :param flow_first:
    :param flow_second:
    :return:
    """
    from scipy import interpolate
    flow_first = generate_absolute_optical_flow(flow_first).permute([0, 3, 1, 2])
    flow_second = generate_absolute_optical_flow(flow_second)
    flow = F.grid_sample(flow_first, flow_second, padding_mode="zeros").permute([0, 2, 3, 1])
    image = F.grid_sample(image, flow, padding_mode="zeros", mode=mode)
    return image

# def warp_image_muti_octave(image, flow_list):
#     """
#     Warp image with multi-octave optical flow.
#     Args:
#         image: image to warp
#         flow_list: multi-octave optical flow
#     Returns:
#         warped image
#     """
#     size = image.size()
#     # resize flow to image size
#     flow_list_resized = [F.interpolate(flow, (size[2], size[3]), mode='bilinear') for flow in flow_list]
#     flow = flow_list_resized[-1]
#     for i in range(1, len(flow_list)):
#         flow += flow_list_resized[-1 - i]
#     return warp_image_relative(image, flow)
