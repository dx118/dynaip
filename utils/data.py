import torch
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

def normalize_imu(acc, ori):
    r"""
    normalize imu w.r.t the root sensor
    """
    acc = acc.view(-1, 6, 3)
    ori = ori.view(-1, 6, 3, 3)
    acc = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(ori[:, 0])
    ori = torch.cat((ori[:, :1], ori[:, :1].transpose(2, 3).matmul(ori[:, 1:])), dim=1)
    data = torch.cat((ori.view(-1, 6, 9), acc), dim=-1)
    return data

def fill_dip_nan(tensor):
    nan_indices = torch.isnan(tensor)
    filled_tensor = tensor.clone()
    for t in range(tensor.size(0)): 
        for i in range(tensor.size(1)):  
            for j in range(tensor.size(2)):  
                if nan_indices[t, i, j]:
                    left_idx = t - 1
                    while left_idx >= 0 and torch.isnan(tensor[left_idx, i, j]):
                        left_idx -= 1
                    left_neighbor_value = tensor[left_idx, i, j] if left_idx >= 0 else 0
                    
                    right_idx = t + 1
                    while right_idx < tensor.size(0) and torch.isnan(tensor[right_idx, i, j]):
                        right_idx += 1
                    right_neighbor_value = tensor[right_idx, i, j] if right_idx < tensor.size(0) else 0

                    filled_tensor[t, i, j] = (left_neighbor_value + right_neighbor_value) / 2
    return filled_tensor


def _syn_acc(v):
    smooth_n = 4
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def interpolate_rotations(rotations: np.ndarray, ts_in, ts_out):
    r"""
    https://github.com/eth-ait/aitviewer/
    Interpolate rotations given at timestamps `ts_in` to timestamps given at `ts_out`. This performs the equivalent
    of cubic interpolation in SO(3).
    :param rotations: A numpy array of rotations of shape (F, N, 3), i.e. rotation vectors.
    :param ts_in: Timestamps corresponding to the given rotations, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    out = []
    for j in range(rotations.shape[1]):
        rs = R.from_rotvec(rotations[:, j])
        spline = RotationSpline(ts_in, rs)
        rs_interp = spline(ts_out).as_rotvec()
        out.append(rs_interp[:, np.newaxis])
    return np.concatenate(out, axis=1)


def resample_rotations(rotations: np.ndarray, fps_in, fps_out):
    r"""
    Resample a motion sequence from `fps_in` to `fps_out`.
    :param rotations: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired frequency of the output sequence.
    :return: A numpy array of shape (F', N, 3) where F is adjusted according to the new fps.
    """
    n_frames = rotations.shape[0]
    assert n_frames > 1, "We need at least two rotations for a resampling to make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_rotations(rotations, ts_in, ts_out)


def interpolate_positions(positions: np.ndarray, ts_in, ts_out):
    r"""
    Interpolate positions given at timestamps `ts_in` to timestamps given at `ts_out` with a cubic spline.
    :param positions: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given positions, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    cs = CubicSpline(ts_in, positions, axis=0)
    new_positions = cs(ts_out)
    return new_positions


def resample_positions(positions: np.ndarray, fps_in, fps_out):
    r"""
    Resample 3D positions from `fps_in` to `fps_out`.
    :param positions: A numpy array of shape (F, ...).
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired output frequency.
    :return: A numpy array of shape (F', ...) where F is adjusted according to the new fps.
    """
    n_frames = positions.shape[0]
    assert n_frames > 1, "Resampling with one data point does not make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_positions(positions, ts_in, ts_out)