import os
import sys

sys.path.append(os.path.abspath('./'))

import torch
import utils.config as cfg
from utils.read_data import read_mvnx, read_xlsx

raw_dir = cfg.raw_dir
extract_dir = cfg.extract_dir
def extract_mvnx():
    datasets = ['andy', 'emokine', 'unipd', 'virginia']
    for dataset in datasets:
        data_folder = os.path.join(raw_dir, dataset)
        mvnx_files = [os.path.relpath(os.path.join(foldername, filename), data_folder)
                    for foldername, _, filenames in os.walk(data_folder)
                    for filename in filenames if filename.endswith('.mvnx')]        
        assert mvnx_files != [], f"No .mvnx files found in {os.path.join(raw_dir, dataset)}"
        for f in mvnx_files:
            f = os.path.join(data_folder, f)
            data = read_mvnx(f)
            if dataset == 'andy':
                f = f.replace('\\', '/').replace('.xsens.mvnx', '.pt')
            else:
                f = f.replace('\\', '/').replace('.mvnx', '.pt')
            print('saving:', f.split('/')[-1])
            dataset = 'virginia_temp' if dataset == 'virginia' else dataset
            out_dir = os.path.join(extract_dir, dataset, f.split('/')[-1])
            os.makedirs(os.path.join(extract_dir, dataset), exist_ok=True)
            torch.save(data, out_dir)        
            
    # we muanually picked a part of data which has no visible drift from original virginia-natural-motion dataset
    # you can visualize the extracted virginia-natural-motion data and select clean clips of your own 
    clip_config = [
        {'name': 'P1_Day_1_1', 'start': [0], 'end': [-1]},
        {'name': 'P1_Day_1_3', 'start': [21800, 35800], 'end': [27000, 71800]},
        {'name': 'P2_Day_1_1', 'start': [0, 82000], 'end': [69000, -1]},
        {'name': 'P3_Day_1_1', 'start': [0], 'end': [50000]},
        {'name': 'P3_Day_1_2', 'start': [0], 'end': [-1]},
        {'name': 'P4_Day_1_1', 'start': [0], 'end': [18000]},
        {'name': 'P4_Day_1_2', 'start': [0], 'end': [43000]},
        {'name': 'P4_Day_1_3', 'start': [0], 'end': [18000]},
        {'name': 'P5_Day_1_1', 'start': [16000], 'end': [34000]},
        {'name': 'P6_Day_2_1', 'start': [80000], 'end': [110000]},
        {'name': 'P10_Day_1_1', 'start': [82000], 'end': [100000]},
        {'name': 'P11_Day_1_2', 'start': [31000, 44000, 206300, 229300], 'end': [33500, 51800, 210300, 238300]},
        {'name': 'P13_Day_2_1', 'start': [0], 'end': [-1]},
        {'name': 'P13_Day_2_2', 'start': [0], 'end': [22500]},
    ]
    
    for d in clip_config:
        name = d['name']
        start_indices = d['start']
        end_indices = d['end']

        data = torch.load(os.path.join(extract_dir, 'virginia_temp', name + '.pt'))
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            out = {'joint': {'orientation': [], 'position': []},
                        'imu': {'free acceleration': [], 'calibrated orientation': []}
                        }     
            out['joint']['orientation'] = data['joint']['orientation'][start:end].float()
            out['joint']['position'] = data['joint']['position'][start:end].float()
            out['imu']['free acceleration'] = data['imu']['free acceleration'][start:end].float()
            out['imu']['calibrated orientation'] = data['imu']['calibrated orientation'][start:end].float()
            print('saving:', '{}_{}'.format(name, i))
            os.makedirs(os.path.join(extract_dir, 'virginia'), exist_ok=True)        
            torch.save(out, os.path.join(extract_dir, 'virginia', '{}_{}.pt'.format(name, i)))         
        
            
def extract_xlsx():
    data_folder = os.path.join(raw_dir, 'cip')
    xlsx_files = [os.path.relpath(os.path.join(foldername, filename), data_folder)
                for foldername, _, filenames in os.walk(data_folder)
                for filename in filenames if filename.endswith('.xlsx')]    
    assert xlsx_files != [], f"No .xlsx files found in {data_folder}"
    for f in xlsx_files:
        f = os.path.join(data_folder, f)
        data = read_xlsx(f)
        f = f.replace('\\', '/').replace('.xlsx', '')
        print('saving:', f.split('/')[-1] + '_' + f.split('/')[5])
        out_dir = os.path.join(extract_dir, 'cip', f.split('/')[-1]) + '_' + f.split('/')[5] + '.pt' # trails, trails_extra_outdoors...
        os.makedirs(os.path.join(extract_dir, 'cip'), exist_ok=True)
        torch.save(data, out_dir)
        
if __name__ == '__main__':
    """extract xsens mocap data from .mvnx and .xlsx files to .pt files"""
    extract_mvnx()
    extract_xlsx()
    print('\n')
    print('Done')
    
    """visualize the extracted data"""
    import articulate as art
    from aitviewer.renderables.skeletons import Skeletons
    from aitviewer.viewer import Viewer
    from utils.skeleton import XsensSkeleton
    
    data_folder = os.path.join(extract_dir, 'virginia')
    extract_files = [os.path.relpath(os.path.join(foldername, filename), data_folder)
                for foldername, _, filenames in os.walk(data_folder)
                for filename in filenames if filename.endswith('.pt')]
    
    data = torch.load(os.path.join(data_folder, extract_files[0]))
    xsens_sk = XsensSkeleton()
    
    # vis joint orientation
    n_frame, n_joints = data['joint']['orientation'].shape[:2]
    orientations = art.math.quaternion_to_rotation_matrix(data['joint']['orientation']).view(n_frame, n_joints, 3, 3)   
    joints = xsens_sk.forward_kinematics(orientations) 
    
    # # vis joint position (root included)
    # joints = data['joint']['position']
    
    skeletion_gt = Skeletons(
                joints.numpy(),
                xsens_sk.connections,
                gui_affine=False,
                radius=0.025,
                name="Xsens GT Skeleton",
            )    

    v = Viewer()
    v.run_animations = True
    v.scene.add(skeletion_gt)
    v.run()     
