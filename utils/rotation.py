import math 
import torch
from scipy import ndimage, spatial

def rotate_vol_around_axis(vol, rot_angle, rot_axis, output_shape=None, order=3): 
    vol_shape = torch.tensor(vol.shape[-3:])
    if output_shape is None:
        output_shape = vol_shape
    # need later for cropping
    crop_offset = [math.floor((vs - cs) / 2) for vs, cs in zip(vol_shape, output_shape)]
    if rot_angle != 0:  
        if not torch.is_tensor(rot_angle):
            rot_angle = torch.tensor(rot_angle)
        rot_angle = torch.deg2rad(rot_angle)
        # convert rotation axis and angle to a 3x3 rotation matrix
        rot_axis = rot_axis.float()
        rot = spatial.transform.Rotation.from_rotvec(rot_angle * (rot_axis/rot_axis.norm()))
        rot_mat = rot.as_matrix()
        # determine offset to rotate around center of volume
        # see https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
        c_in = 0.5 * (vol_shape - torch.ones(3)).float().numpy()  # -1 because indexing starts at 0
        offset = c_in - rot_mat@c_in
        # Apply the rotation using affine_transform
        vol = torch.tensor(
            ndimage.affine_transform(
                vol, 
                matrix=rot_mat,
                offset=offset, 
                order=order
            ),
            device=vol.device, 
            dtype=vol.dtype
        )
    vol = vol[crop_offset[0]:crop_offset[0]+output_shape[0], crop_offset[1]:crop_offset[1]+output_shape[1], crop_offset[2]:crop_offset[2]+output_shape[2]]
    return vol