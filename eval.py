import glob
import os

import articulate as art
import torch
import utils.config as cfg
from model.model import Poser
from utils.evaluator import Evaluator
from tqdm import tqdm

body_model = art.ParametricModel(cfg.smpl_m, device='cpu')  

def evaluate(net, dataset: str, pose_evaluator=Evaluator()):
    total_error = []
    for f in tqdm(glob.glob(os.path.join(cfg.work_dir, 'test', dataset, '*.pt'))):
        data = torch.load(f)
        # prepare imu and initial states of the first frame
        imu = data['imu']['imu'].cuda()
        if dataset == 'dip':
            vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
            local_smpl = data['joint']['full smpl pose']
            glb_gt_smpl, _ = body_model.forward_kinematics(local_smpl, calc_mesh=False) 
        else:
            vel_mask = torch.tensor([0, 6, 14, 10, 21, 17])
            glb_gt_xsens = data['joint']['full xsens pose']
            
        v_init = data['joint']['velocity'][:1, vel_mask].float().cuda()
        pose = data['joint']['orientation']
        pose = pose.view(pose.shape[0], -1, 6)[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]]
        p_init = pose[:1].cuda()        
        
        glb_full_pose_xsens, glb_full_pose_smpl = net.predict(imu, v_init, p_init)
        
        if dataset == 'dip':
            err = pose_evaluator.eval_smpl(glb_full_pose_smpl, glb_gt_smpl)
        else:
            err = pose_evaluator.eval_xsens(glb_full_pose_xsens, glb_gt_xsens)
        total_error.append(err)
    total_error = torch.stack(total_error).mean(dim=0)
    return total_error
    
if __name__ == '__main__':
    net = Poser().cuda()
    net.load_state_dict(torch.load(cfg.weight, map_location='cuda:0')) # DynaIP in paper
    # net.load_state_dict(torch.load(cfg.weight_s, map_location='cuda:0')) # DynaIP* in paper
    net.eval()
    
    log = []
    datasets = ['dip', 'andy', 'unipd', 'cip', 'virginia']


    for ds in datasets:
        total_error = evaluate(net, ds)
        log.append([ds, total_error])

    print('-' * 75)
    print(f'{" " * 10:^10} | {"SIP Err":^18} | {"Global Angle Err":^18} | {"Joint Position Err":^18}')    
    for (ds, err) in log:
        print('-' * 75)
        print(f'{ds:^10} | {"{}".format(round(err[0].item(), 2)):^18} | {"{}".format(round(err[1].item(), 2)):^18} | {"{}".format(round(err[2].item(), 2)):^18}')