import os
import sys

sys.path.append(os.path.abspath('./'))

import glob

import torch
import utils.config as cfg
from torch.utils.data import Dataset
from tqdm import tqdm


class MotionDataset(Dataset):
    def __init__(self, datasets=['unipd', 'cip', 'andy', 'emokine', 'virginia'], seq_len=300, device='cuda:0'):
        super(MotionDataset, self).__init__()
        self.train_dir = os.path.join(cfg.work_dir, 'train')
        self.datasets = datasets    
        self.seq_len = seq_len
        self.data = {'imu': [], 'pose': [], 'p_init': [], 'velocity': [],
                     'v_init': []}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prepare_data()
        print("{} samples in total.".format(len(self.data['imu'])))
        
    def __len__(self):
        return len(self.data['imu'])
    
    def prepare_data(self):
        r"""
        joint_mask: ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'LeftLowerLeg', 'RightLowerLeg', 'L3', 
         'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm', 
         'RightUpperArm', 'LeftForeArm', 'RightForeArm'] 
        """
        for dataset in self.datasets:
            temp_data = {'imu': [], 'velocity': [], 'pose': []}
            for motion_name in tqdm(glob.glob(os.path.join(self.train_dir, dataset, '*.pt'))):
                data = torch.load(motion_name)
                
                imu = data['imu']['imu'].view(-1, 6, 12)
                temp_data['imu'].extend(torch.split(imu.view(imu.shape[0], 6, 12), self.seq_len))

                # 'Pelvis' "head" "left_wrist" "right_wrist" "left_ankle" "right_ankle"
                if data['joint']['velocity'].shape[1] == 24: # smpl
                    vel_mask = torch.tensor([0, 15, 20, 21, 7, 8]) 
                else:
                    vel_mask = torch.tensor([0, 6, 14, 10, 21, 17])  # xsens
                
                velocity = data['joint']['velocity'][:, vel_mask].float()                
                temp_data['velocity'].extend(torch.split(velocity.view(velocity.shape[0], -1, 3), self.seq_len))
                # Discard pelvis, head, wrists and ankles, which have direct imu-readings
                pose = pose.view(pose.shape[0], -1, 6)[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]]
                temp_data['pose'].extend(torch.split(pose, self.seq_len))
                              
             
            self.data['imu'].extend(temp_data['imu'])
            self.data['velocity'].extend(temp_data['velocity'])
            self.data['pose'].extend(temp_data['pose'])

            for i in range(len(temp_data['imu'])):
                self.data['v_init'].append(temp_data['velocity'][i][:1])
                self.data['p_init'].append(temp_data['pose'][i][:1])

    def __getitem__(self, index):
        imu = self.data['imu'][index].to(self.device)
        vel = self.data['velocity'][index].to(self.device)
        v_init = self.data['v_init'][index].to(self.device)
        pose = self.data['pose'][index].to(self.device)
        p_init = self.data['p_init'][index].to(self.device)
        return imu, vel, pose, v_init, p_init

    
def collate_fn(batch):
    imu = [item[0] for item in batch]
    vel = [item[1] for item in batch]
    pose = [item[2] for item in batch]
    v_init = torch.cat([item[3] for item in batch], dim=0)
    p_init = torch.cat([item[4] for item in batch], dim=0)
    return imu, vel, pose, v_init, p_init

