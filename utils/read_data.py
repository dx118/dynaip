r"""
    Reader for Xsens .mvnx file. modified from https://github.com/Xinyu-Yi/PIP
"""


__all__ = ['read_mvnx', 'read_xlsx']


import xml.etree.ElementTree as ET
import torch
import glob
import numpy as np

def quaternion_product(q1, q2):
    r"""
    Quaternion in w, x, y, z.

    :param q1: Tensor [N, 4].
    :param q2: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    w1, xyz1 = q1.reshape(-1, 4)[:, :1], q1.reshape(-1, 4)[:, 1:]
    w2, xyz2 = q2.reshape(-1, 4)[:, :1], q2.reshape(-1, 4)[:, 1:]
    xyz = torch.cross(xyz1, xyz2) + w1 * xyz2 + w2 * xyz1
    w = w1 * w2 - (xyz1 * xyz2).sum(dim=1, keepdim=True)
    q = torch.cat((w, xyz), dim=1).view_as(q1)
    return q


def quaternion_inverse(q):
    r"""
    Quaternion in w, x, y, z.

    :param q: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    invq = q.clone().reshape(-1, 4)
    invq[:, 1:].neg_()
    return invq.view_as(q)


def quaternion_normalize(q):
    r"""
    Quaternion in w, x, y, z.

    :param q: Tensor [N, 4].
    :return: Tensor [N, 4].
    """
    q_normalized = q.reshape(-1, 4) / q.reshape(-1, 4).norm(dim=1, keepdim=True)
    return q_normalized.view_as(q)


def read_mvnx(file: str):
    r"""
    Parse a mvnx file. All measurements are converted into the SMPL coordinate frame. The result is saved in a dict:

    return dict:
        framerate: int                                 --- fps

        timestamp ms: Tensor[nframes]                  --- timestamp in ms

        center of mass: Tensor[nframes, 3]             --- CoM position

        joint:
            name: List[njoints]                        --- joint order (name)
            <others>: Tensor[nframes, njoints, ndim]   --- other joint properties computed by Xsens

        imu:
            name: List[nimus]                          --- imu order (name)
            <others>: Tensor[nframes, nimus, ndim]     --- other imu measurements


        tpose: ...                                     --- tpose information

    :param file: Xsens file `*.mvnx`.
    :return: The parsed dict.
    """
    tree = ET.parse(file)

    # read framerate
    frameRate = int(tree.getroot()[2].attrib['frameRate'])

    # read joint order
    segments = tree.getroot()[2][1]
    n_joints = len(segments)
    joints = []
    for i in range(n_joints):
        assert int(segments[i].attrib['id']) == i + 1
        joints.append(segments[i].attrib['label'])

    # read imu order
    sensors = tree.getroot()[2][2]
    n_imus = len(sensors)
    imus = []
    for i in range(n_imus):
        imus.append(sensors[i].attrib['label'])



    # read frames
    frames = tree.getroot()[2][-1]
    data = {'framerate': frameRate,
            'timestamp ms': [],
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            'tpose': {}
            }    
    
    if frameRate != 60:
        step = int(frameRate // 60)
    else:
        step = 1


    for i in range(len(frames)):
        if frames[i].attrib['type'] in ['identity', 'tpose', 'tpose-isb']: # virginia
            data['tpose'][frames[i].attrib['type']] = {
                'orientation': torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4),
                'position': torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
            }
            continue
       
        elif ('index' in frames[i].attrib) and (frames[i].attrib['index'] == ''): # unipd 
            data['tpose'][frames[i].attrib['type']] = {
                'orientation': torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4),
                'position': torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
            }
            continue
        
        # assert frames[i].attrib['type'] == 'normal' and \
        #        int(frames[i].attrib['index']) == len(data['timestamp ms'])

        orientation = torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4)
        position = torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
        sensorFreeAcceleration = torch.tensor([float(_) for _ in frames[i][7].text.split(' ')]).view(n_imus, 3)
        try:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][9].text.split(' ')]).view(n_imus, 4)
        except:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][8].text.split(' ')]).view(n_imus, 4)

        data['timestamp ms'].append(int(frames[i].attrib['time']))
        data['joint']['orientation'].append(orientation)
        data['joint']['position'].append(position)
        data['imu']['free acceleration'].append(sensorFreeAcceleration)
        data['imu']['orientation'].append(sensorOrientation)

    data['timestamp ms'] = torch.tensor(data['timestamp ms'])
    for k, v in data['joint'].items():
        data['joint'][k] = torch.stack(v)
    for k, v in data['imu'].items():
        data['imu'][k] = torch.stack(v)

    data['joint']['name'] = joints
    data['imu']['name'] = imus


    # to smpl coordinate frame
    def convert_quaternion_(q):
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_pose = R mvnx_pose R^T
        """
        oldq = q.view(-1, 4).clone()
        q.view(-1, 4)[:, 1] = oldq[:, 2]
        q.view(-1, 4)[:, 2] = oldq[:, 3]
        q.view(-1, 4)[:, 3] = oldq[:, 1]

    def convert_point_(p): 
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_point = R mvnx_point
        """
        oldp = p.view(-1, 3).clone()
        p.view(-1, 3)[:, 0] = oldp[:, 1]
        p.view(-1, 3)[:, 1] = oldp[:, 2]
        p.view(-1, 3)[:, 2] = oldp[:, 0]

    convert_quaternion_(data['joint']['orientation'])
    convert_point_(data['joint']['position'])
    convert_quaternion_(data['imu']['orientation'])
    convert_point_(data['imu']['free acceleration'])
    convert_quaternion_(data['tpose']['identity']['orientation'])
    convert_quaternion_(data['tpose']['tpose']['orientation'])
    convert_quaternion_(data['tpose']['tpose-isb']['orientation'])
    convert_point_(data['tpose']['identity']['position'])
    convert_point_(data['tpose']['tpose']['position'])
    convert_point_(data['tpose']['tpose-isb']['position'])
    
    if step != 1:
        data['joint']['orientation'] = data['joint']['orientation'][::step].clone()
        data['joint']['position'] = data['joint']['position'][::step].clone()
        data['imu']['free acceleration'] = data['imu']['free acceleration'][::step].clone()
        data['imu']['orientation'] = data['imu']['orientation'][::step].clone()

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]
    q_off = quaternion_product(quaternion_inverse(data['imu']['orientation'][:n_frames_for_calibration]), data['joint']['orientation'][:n_frames_for_calibration, imu_idx])
    ds = q_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        q_off[:, i] = q_off[:, i] * q_off[:, i, d:d+1].sign()
    q_off = quaternion_normalize(quaternion_normalize(q_off).mean(dim=0))
    data['imu']['calibrated orientation'] = quaternion_product(data['imu']['orientation'], q_off.repeat(data['imu']['orientation'].shape[0], 1, 1)).clone()


        
    print('file_name:', file)          
    print('total frames:', len(frames), 'step:', step)
          
    return data




def read_xlsx(xsens_file_path):
    r"""
    Extracts data from Xsens .xlsx file. modified from https://github.com/ManuelPalermo/HumanInertialPose
    """
    
    # to smpl coordinate frame
    def convert_quaternion_(q):
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_pose = R mvnx_pose R^T
        """
        q[:, :, [1, 2, 3]] = q[:, :, [2, 3, 1]]
        return q

    def convert_point_(p): 
        r""" inplace convert
            R = [[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]]
            smpl_point = R mvnx_point
        """
        p[:, :, [0, 1, 2]] = p[:, :, [1, 2, 0]]
        return p  

    # extract xsens general data (.xlsx)
    import pandas as pd

    pos3s_com, segments_pos3d, segments_quat, \
        imus_ori, imus_free_acc = pd.read_excel(
            xsens_file_path,
            sheet_name=["Center of Mass",
                        "Segment Position",               # positions of joints in 3d space
                        "Segment Orientation - Quat",     # segment global orientation 
                        "Sensor Orientation - Quat",      # sensor orientation
                        "Sensor Free Acceleration",       # sensor free acceleration (accelerometer data without gravity vector)
                        ],
            index_col=0
        ).values()

    data = {'framerate': 60.,
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            }

    # add dim (S, [1], 3)  +  ignore com_vel / com_accel
    pos3s_com = np.expand_dims(pos3s_com.values, axis=1)[..., [0, 1, 2]]
    n_samples = len(pos3s_com)

    # assumes a perfect sampling freq of 60hz
    timestamps = np.arange(1, n_samples + 1) * (1 / 60.)

    segments_pos3d = segments_pos3d.values.reshape(n_samples, -1, 3)
    segments_quat = segments_quat.values.reshape(n_samples, -1, 4)
    imus_free_acc = imus_free_acc.values.reshape(n_samples, -1, 3)
    imus_ori = imus_ori.values.reshape(n_samples, -1, 4)
    mask = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21])
    imus_ori = imus_ori[:, mask, :]
    imus_free_acc = imus_free_acc[:, mask, :]

    data['joint']['orientation'] = torch.tensor(segments_quat.astype(np.float32)).clone()
    data['joint']['position'] = torch.tensor(segments_pos3d.astype(np.float32)).clone()
    data['imu']['orientation'] = torch.tensor(imus_ori.astype(np.float32)).clone()
    data['imu']['free acceleration'] = torch.tensor(imus_free_acc.astype(np.float32)).clone()
    

    data['joint']['orientation'] = convert_quaternion_(data['joint']['orientation'])
    data['joint']['position'] = convert_point_(data['joint']['position'])
    data['imu']['orientation'] = convert_quaternion_(data['imu']['orientation'])
    data['imu']['free acceleration'] = convert_point_(data['imu']['free acceleration'])

    data['joint']['name'] = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'RightShoulder', 'RightUpperArm',
                            'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                            'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe', 'LeftUpperLeg', 'LeftLowerLeg',
                            'LeftFoot', 'LeftToe']
    data['imu']['name'] = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                        'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg',
                        'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
        
    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]
    q_off = quaternion_product(quaternion_inverse(data['imu']['orientation'][:n_frames_for_calibration]), data['joint']['orientation'][:n_frames_for_calibration, imu_idx])
    ds = q_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        q_off[:, i] = q_off[:, i] * q_off[:, i, d:d+1].sign()
    q_off = quaternion_normalize(quaternion_normalize(q_off).mean(dim=0))
    data['imu']['calibrated orientation'] = quaternion_product(data['imu']['orientation'], q_off.repeat(data['imu']['orientation'].shape[0], 1, 1))
    
    return data