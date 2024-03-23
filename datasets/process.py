import os
import sys

sys.path.append(os.path.abspath('./'))

import glob

import articulate as art
import numpy as np
import torch
import utils.config as cfg
from tqdm import tqdm
from utils.data import fill_dip_nan, normalize_imu

raw_dir = cfg.raw_dir
extract_dir = cfg.extract_dir
work_dir = cfg.work_dir
split_dir = cfg.split_dir
smpl_m = cfg.smpl_m
dip_dir = os.path.join(cfg.raw_dir, 'dip')
out_dir = os.path.join(cfg.raw_dir, 'dip_trans')


def process_xsens():
    r"""
    imu_mask: [Pelvis, LeftKnee, RightKnee, Head, LeftElbow, RightElbow]
    
    joint_mask: ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'LeftLowerLeg', 'RightLowerLeg', 'L3', 
                 'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm', 
                 'RightUpperArm', 'LeftForeArm', 'RightForeArm']                                
    """     

    imu_mask = torch.tensor([0, 15, 12, 2, 9, 5])
    joint_mask = torch.tensor([19, 15, 1, 20, 16, 2, 3, 4, 5, 11, 7, 6, 12, 8, 13, 9])

    print('\n')
    infos = os.listdir(os.path.join(split_dir, 'xsens'))
    for info_name in infos:
        dataset, phase = info_name.split('.')[0].split('_')
        with open(os.path.join(split_dir, 'xsens', info_name), 'r') as file:
            l = file.read().splitlines()
        print('processing {}_{}...'.format(dataset, phase))
        for motion_name in tqdm(l):
            temp_data = torch.load(os.path.join(extract_dir, dataset, motion_name))
       
            out_data = {'joint': {'orientation': [], 'velocity': []},
                        'imu': {'imu': []},
                        }                  
            
            # normalize and to r6d
            glb_pose = art.math.quaternion_to_rotation_matrix(temp_data['joint']['orientation']).view(-1, 23, 3, 3)
            out_data['joint']['full xsens pose'] = glb_pose # glb gt
            
            glb_pose_norm = glb_pose[:, :1].transpose(2, 3).matmul(glb_pose)
            glb_pose_norm = glb_pose_norm[:, joint_mask].view(-1, len(joint_mask), 3, 3)[:, :, :, :2].transpose(2, 3).clone().flatten(1) 
            out_data['joint']['orientation'] = glb_pose_norm 
                
            acc = temp_data['imu']['free acceleration'][:, imu_mask].view(-1, 6, 3)
            ori = art.math.quaternion_to_rotation_matrix(temp_data['imu']['calibrated orientation'][:, imu_mask]).view(-1, 6, 9)
            out_data['imu']['imu'] = normalize_imu(acc, ori)
            
            # calculate velocity and normalize w.r.t. root orientation
            gt_position = temp_data['joint']['position'] # N 23 3
            # remove horizon movement
            gt_position[:, :, 0] = gt_position[:, :, 0] - gt_position[:, :1, 0]
            gt_position[:, :, 2] = gt_position[:, :, 2] - gt_position[:, :1, 2]
            
            velocity = (gt_position[1:] - gt_position[:-1]) * 60.0
            velocity = torch.cat((velocity[:1], velocity), dim=0)
            velocity = torch.cat((velocity[:, :1], velocity[:, 1:] - velocity[:, :1]), dim=1).bmm(ori[:, 0].view(-1, 3, 3))

            out_data['joint']['velocity'] = velocity
            out_data['joint']['position'] = gt_position.bmm(ori[:, 0].view(-1, 3, 3))
            
            out_dir = os.path.join(work_dir, phase, dataset, motion_name)
            os.makedirs(os.path.join(work_dir, phase, dataset), exist_ok=True)
            torch.save(out_data, out_dir)         

def generate_dip_trans():
    import pickle
    split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10']
    for subject_name in tqdm(split):
        for motion_name in os.listdir(os.path.join(dip_dir, subject_name)):
            out_data = {
                'imu_acc': [],
                'imu_ori': [],
                'pose': [],
                'trans': []
                    }
            
            path = os.path.join(dip_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc']).float()
            ori = torch.from_numpy(data['imu_ori']).float()        
            pose_aa = torch.from_numpy(data['gt']).float()
            
            pose = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)
            body_model = art.ParametricModel(smpl_m, device='cpu')

            lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
            lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
                
            j, _ = body_model.get_zero_pose_joint_and_vertex()
            b = art.math.joint_position_to_bone_vector(j[lower_body].unsqueeze(0),
                                                        lower_body_parent).squeeze(0)
            bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
            b = bone_orientation * bone_length
            b[:3] = 0
            floor_y = j[10:12, 1].min().item()

            j = body_model.forward_kinematics(pose=pose, calc_mesh=False)[1]

            trans = torch.zeros(j.shape[0], 3)

            # force lowest foot to the floor
            for i in range(j.shape[0]):
                current_foot_y = j[i, [10, 11], 1].min().item()
                if current_foot_y > floor_y:
                    trans[i, 1] = floor_y - current_foot_y
            
            out_data['imu_acc'] = acc
            out_data['imu_ori'] = ori
            out_data['pose'] = pose_aa
            out_data['trans'] = trans.float()
            
            os.makedirs(os.path.join(raw_dir, 'dip_trans'), exist_ok=True)
            torch.save(out_data, os.path.join(raw_dir, 'dip_trans', subject_name + '_' + motion_name.replace('.pkl', '.pt')))    


def process_dipimu():
    r"""
    imu_mask: [Pelvis, LeftKnee, RightKnee, Head, LeftElbow, RightElbow]
    joint_mask: ['Left_hip', 'Right_hip', 'Spine1', 'Left_knee', 'Right_knee', 'Spine1', Spine2', 
                 'Spine3', 'Neck', 'Left_collar', 'Right_collar', 'Head', 'Left_shoulder', 
                 'Right_shoulder', 'Left_elbow', 'Right_elbow'] duplicate one spine(e.g. spine1) to match xsens                 
    """
    print('\n')
    print('generating dip-imu pseudo vertical translation...')
    generate_dip_trans()
    
    print('processing dip-imu...')
    imu_mask = torch.tensor([2, 11, 12, 0, 7, 8])
    joint_mask = torch.tensor([1, 2, 3, 4, 5, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19])
    infos = os.listdir(os.path.join(split_dir, 'dip'))
    for info_name in infos:
        _, phase = info_name.split('.')[0].split('_')
        with open(os.path.join(split_dir, 'dip', info_name), 'r') as file:
            l = file.read().splitlines()
        for motion_name in tqdm(l):
            path = os.path.join(raw_dir, 'dip_trans', motion_name)
            data = torch.load(path)
            
            acc = data['imu_acc'][:, imu_mask]
            ori = data['imu_ori'][:, imu_mask]
            pose = data['pose']
            trans = data['trans']

            out_data = {'joint': {'orientation': [], 'velocity': [], 'position': []},
                        'imu': {'imu': []},
                        }                       
            
            # fill nan with nearest neighbors
            if True in torch.isnan(acc):
                acc = fill_dip_nan(acc)
            if True in torch.isnan(ori):
                ori = fill_dip_nan(ori.view(-1, 6, 9))
                
            body_model = art.ParametricModel(smpl_m, device='cpu')  
            p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)            
            glb_pose, gt_position  = body_model.forward_kinematics(pose=p, tran=trans, calc_mesh=False)        
            
            # normalize and to r6d
            glb_pose = glb_pose[:, :1].transpose(2, 3).matmul(glb_pose)
            glb_pose_norm = glb_pose[:, joint_mask].view(-1, len(joint_mask), 3, 3)[:, :, :, :2].transpose(2, 3).clone().flatten(1)
            
            acc, ori, glb_pose_norm, gt_position = acc[6:-6], ori[6:-6], glb_pose_norm[6:-6], gt_position[6:-6]
            p = p[6:-6]
     
            gt_position[:, :, 0] = gt_position[:, :, 0] - gt_position[:, :1, 0]
            gt_position[:, :, 2] = gt_position[:, :, 2] - gt_position[:, :1, 2]  
            out_data['joint']['position'] = gt_position.bmm(ori[:, 0].view(-1, 3, 3))
          
            velocity = (gt_position[1:] - gt_position[:-1]) * 60
            velocity = torch.cat((velocity[:1], velocity), dim=0)
            velocity = torch.cat((velocity[:, :1], velocity[:, 1:] - velocity[:, :1]), dim=1).bmm(ori[:, 0].view(-1, 3, 3))            
            
            out_data['joint']['velocity'] = velocity # N, 24, 3, 3    
            out_data['joint']['orientation'] = glb_pose_norm # N 90
            out_data['imu']['imu'] = normalize_imu(acc, ori)
            out_data['joint']['full smpl pose'] = p # local gt

            out_dir = os.path.join(work_dir, phase, 'dip', motion_name)
            os.makedirs(os.path.join(work_dir, phase, 'dip'), exist_ok=True)
            torch.save(out_data, out_dir)     

if __name__ == '__main__':
    process_xsens()
    process_dipimu()