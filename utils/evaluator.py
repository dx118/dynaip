import torch
import articulate as art
from articulate.math.angular import radian_to_degree, angle_between
from utils.skeleton import XsensSkeleton
import utils.config as cfg

class Evaluator:
    def __init__(self, sip_xsens=torch.tensor([8, 12, 15, 19]), sip_smpl=torch.tensor([1, 2, 16, 17])):
        self.names = ['SIP Error (deg)', 'Global Angle Error (deg)']
        self.body_model = art.ParametricModel(cfg.smpl_m, device='cpu')
        self.xsens_sk = XsensSkeleton()
        self.sip_xsens = sip_xsens
        self.sip_smpl = sip_smpl

    def eval_xsens(self, pose_p, pose_t):
        r"""
        Args:
            pose_p :xsens prediction rotation matrix that can reshape to [num_frame, 23, 3, 3].
            pose_t :xsens gt rotation matrix that can reshape to [num_frame, 23, 3, 3].  
        Returns:    
            gae: Global Angular Error (deg)
            mgae: SIP Error (deg)
            je: Joint Position Error (cm)
        """
        pose_p = pose_p.clone().view(-1, 23, 3, 3)
        pose_t = pose_t.clone().view(-1, 23, 3, 3)
        ignored_joint_mask = torch.tensor([0, 10, 14, 17, 18, 21, 22])
        # ignored_joint_mask = torch.tensor([0, 2, 10, 14, 17, 18, 21, 22]) # For PIP and TIP, do not calc L3 error
        
        # replace ignored joint with ground truth global rotation
        pose_p[:, ignored_joint_mask] = pose_t[:, ignored_joint_mask]
        gae = radian_to_degree(angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))
        mgae = radian_to_degree(angle_between(pose_p[:, self.sip_xsens], \
                pose_t[:, self.sip_xsens]).view(pose_p.shape[0], -1))

        # since we did not extract each skeleton offsets, we use forward kinematics
        # and mean skeleton to calculate joint position error
        joint_p = self.xsens_sk.forward_kinematics(pose_p)
        joint_t = self.xsens_sk.forward_kinematics(pose_t)
        je = (joint_p - joint_t).norm(dim=2)
        return torch.stack([mgae.mean(), gae.mean(), je.mean() * 100.0])     
    
    def eval_smpl(self, pose_p, pose_t):
        r"""
        Args:
            pose_p :smpl prediction rotation matrix that can reshape to [num_frame, 24, 3, 3].
            pose_t :smpl gt rotation matrix that can reshape to [num_frame, 24, 3, 3].
            we get smpl prediction by assigning corresponding xsens joints to smpl joints.   
        Returns:    
            gae: Global Angular Error (deg)
            mgae: SIP Error (deg)
            je: Joint Position Error (cm)
            ve: Vertex Position Error (cm)
        """
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])   
        pose_p[:, ignored_joint_mask] = pose_t[:, ignored_joint_mask]
        gae = radian_to_degree(
            angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))
        
        mgae = radian_to_degree(
            angle_between(pose_p[:, self.sip_smpl], \
                pose_t[:, self.sip_smpl]).view(pose_p.shape[0], -1))
        
        pose_p_local = self.body_model.inverse_kinematics_R(pose_p).view(pose_p.shape[0], 24, 3, 3)
        pose_t_local = self.body_model.inverse_kinematics_R(pose_t).view(pose_t.shape[0], 24, 3, 3)
        _, joint_p, vertex_p = self.body_model.forward_kinematics(pose=pose_p_local, calc_mesh=True)
        _, joint_t, vertex_t = self.body_model.forward_kinematics(pose=pose_t_local, calc_mesh=True)
        
        je = (joint_p - joint_t).norm(dim=2)
        ve = (vertex_p - vertex_t).norm(dim=2)
        return torch.stack([mgae.mean(), gae.mean(), je.mean() * 100.0, ve.mean() * 100.0])     
    

class PIPEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._base_motion_loss_fn = art.FullMotionEvaluator(cfg.smpl_m,
                                                            joint_mask=torch.tensor([1, 2, 16, 17]), device=device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 1000])