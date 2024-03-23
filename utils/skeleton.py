r"""
Modified from https://github.com/ARLab-VT/VT-Natural-Motion-Processing
"""
import torch

class Skeleton:
    """Base skeleton and forward kinematics."""
    def __init__(self):
        self.topology = []
        self.offsets = []

    def forward_kinematics(self, orientations, to_smpl=True, to_mvnx=True):   
        r"""
        Forward kinematic for a batch of orientations, in rotation matrix.
        :param orientations: global orientation in smpl frame, predicted by the network,
                             must transform back first.
        :param to_smpl: transform bvh positions to smpl frame, default to True.
        :return: positions in smpl frame.
        """
        device = orientations.device
        if to_mvnx:
            G = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], \
                dtype=torch.float32).repeat(orientations.shape[0], 1, 1).unsqueeze(1).to(device)             
            orientations = G.transpose(2, 3).matmul(orientations).matmul(G)                   
                
        positions = torch.zeros([len(self.offsets), orientations.shape[0], 3], dtype=torch.float32).to(device)

        for i, parent_indices in enumerate(self.topology):
            if parent_indices == -1:
                continue
            else:
                x_B = self.offsets[i].to(device)
                x_B = x_B.view(1, -1, 1).repeat(orientations.shape[0], 1, 1)

                R_GB = orientations[:, parent_indices]
                positions[i] = (positions[parent_indices].to(device) + R_GB.bmm(x_B).squeeze(2))

        if to_smpl:
            R = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32).to(device)
            positions = R.matmul(positions.permute(1, 2, 0).contiguous()).transpose(1, 2)
            return positions
        else:
            return positions.permute(1, 0, 2)
            
class XsensSkeleton(Skeleton):
    """Skeleton defination and forward kinematics for Xsens."""

    def __init__(self):
        r"""
        Initialize the skeleton, using segment lengths from Andy dataset,
        extracted from bvh files.
        
        segments = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
                    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
                    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
                    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
                    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe"]
                    
        parents = [None, "Pelvis", "L5", "L3", "T12", "T8", "Neck",
                   "T8", "RightShoulder", "RightUpperArm", "RightForeArm",
                   "T8", "LeftShoulder", "LeftUpperArm", "LeftForeArm",
                   "Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot",
                   "Pelvis", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot"]    
        """

        self.offsets =  torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                                        [-1.1913e-04,  0.0000e+00,  1.0651e-01],
                                        [ 2.1639e-04,  0.0000e+00,  1.0562e-01],
                                        [ 0.0000e+00,  0.0000e+00,  9.5423e-02],
                                        [ 0.0000e+00,  0.0000e+00,  9.5423e-02],
                                        [ 0.0000e+00,  0.0000e+00,  1.3356e-01],
                                        [ 6.4870e-04,  0.0000e+00,  8.7248e-02],
                                        [ 0.0000e+00, -2.9223e-02,  7.4656e-02],
                                        [ 0.0000e+00, -1.4086e-01,  0.0000e+00],
                                        [ 0.0000e+00, -2.9541e-01,  0.0000e+00],
                                        [ 4.4812e-05, -2.4263e-01,  0.0000e+00], # 10 RightHand
                                        [ 0.0000e+00,  2.9223e-02,  7.4656e-02],
                                        [ 0.0000e+00,  1.4086e-01,  0.0000e+00],
                                        [ 0.0000e+00,  2.9541e-01,  0.0000e+00],
                                        [ 4.4812e-05,  2.4263e-01,  0.0000e+00], # 14 LeftHand
                                        [ 5.9564e-05, -8.7448e-02,  5.8373e-04],
                                        [ 4.6875e-05,  0.0000e+00, -4.6706e-01],
                                        [-1.3225e-04,  0.0000e+00, -4.1930e-01], # 17 RightLowerLeg
                                        [ 1.6630e-01,  0.0000e+00, -1.0138e-01], # 18 RightFoot
                                        [ 5.9564e-05,  8.7448e-02,  5.8373e-04],
                                        [ 4.6875e-05,  0.0000e+00, -4.6706e-01],
                                        [-1.3225e-04,  0.0000e+00, -4.1930e-01], # 21 LeftLowerLeg
                                        [ 1.6630e-01,  0.0000e+00, -1.0138e-01]]) # 22 LeftFoot
                

        self.topology = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15,
                        16, 17, 0, 19, 20, 21]
        self.connections = []
        for i, pi in enumerate(self.topology):
            if pi >= 0:
                self.connections.append([pi, i])             
                                               
    def forward_kinematics(self, orientation, to_smpl=True, to_mvnx=True):
        positions = super().forward_kinematics(orientation, to_smpl, to_mvnx)
        return positions


class SMPLSkeleton(Skeleton):
    """Skeleton defination and forward kinematics for SMPL."""

    def __init__(self):
        r"""
        Initialize the skeleton, using segment lengths standard smpl skeleton (beta=0).
        Additionally normalized with body height.
        
        segments = ['Pelvis', 'Left_hip', 'Left_knee', 'Left_ankle', 'Left_foot', 
          'Right_hip', 'Right_knee', 'Right_ankle', 'Right_foot',
          'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
          'Left_collar', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Left_palm',
          'Right_collar', 'Right_shoulder', 'Right_elbow', 'Right_wrist', 'Right_palm']
          
        parents = [None, 'Pelvis', 'Left_hip', 'Left_knee', 'Left_ankle',
                    'Pelvis', 'Right_hip', 'Right_knee', 'Right_ankle', 'Pelvis',
                    'Spine1', 'Spine2', 'Spine3', 'Neck', 'Spine3',
                    'Left_collar', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Spine3',
                    'Right_collar', 'Right_shoulder', 'Right_elbow', 'Right_wrist']  
        """
        self.offsets =  torch.tensor([[0.0000,  0.0000,  0.0000],
                        [-0.0112,  0.0372, -0.0522],
                        [ 0.0051,  0.0276, -0.2453],
                        [-0.0238, -0.0094, -0.2710],
                        [ 0.0775,  0.0261, -0.0383],
                        [-0.0086, -0.0383, -0.0575],
                        [-0.0031, -0.0275, -0.2436],
                        [-0.0219,  0.0121, -0.2666],
                        [ 0.0827, -0.0221, -0.0394],
                        [-0.0244,  0.0028,  0.0790],
                        [ 0.0170,  0.0028,  0.0876],
                        [ 0.0018, -0.0014,  0.0356],
                        [-0.0212, -0.0085,  0.1343],
                        [ 0.0320,  0.0064,  0.0565],
                        [-0.0120,  0.0455,  0.0724],
                        [-0.0121,  0.0780,  0.0287],
                        [-0.0146,  0.1621, -0.0099],
                        [-0.0047,  0.1687,  0.0081],
                        [-0.0099,  0.0550, -0.0068],
                        [-0.0150, -0.0527,  0.0714],
                        [-0.0054, -0.0719,  0.0297],
                        [-0.0198, -0.1651, -0.0091],
                        [-0.0038, -0.1708,  0.0043],
                        [-0.0064, -0.0563, -0.0055]])

        self.topology = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15,
                        16, 17, 11, 19, 20, 21, 22]
                    
        self.connections = []
        for i, pi in enumerate(self.topology):
            if pi >= 0:
                self.connections.append([pi, i])                  
                                               
    def forward_kinematics(self, orientation, to_smpl=True, to_mvnx=True):
        positions = super().forward_kinematics(orientation, to_smpl, to_mvnx)
        return positions
    
 

class SMPLSkeletonNohand(Skeleton):
    """Skeleton defination and forward kinematics for SMPL."""

    def __init__(self):
        r"""
        Initialize the skeleton, using segment lengths standard smpl skeleton (beta=0)
        
        segments = ['Pelvis', 'Left_hip', 'Left_knee', 'Left_ankle', 'Left_foot', 
          'Right_hip', 'Right_knee', 'Right_ankle', 'Right_foot',
          'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
          'Left_collar', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Left_palm',
          'Right_collar', 'Right_shoulder', 'Right_elbow', 'Right_wrist', 'Right_palm']
          
        parents = [None, 'Pelvis', 'Left_hip', 'Left_knee', 'Left_ankle',
                    'Pelvis', 'Right_hip', 'Right_knee', 'Right_ankle', 'Pelvis',
                    'Spine1', 'Spine2', 'Spine3', 'Neck', 'Spine3',
                    'Left_collar', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Spine3',
                    'Right_collar', 'Right_shoulder', 'Right_elbow', 'Right_wrist']  
        """
        self.offsets =  torch.tensor([[0.0000,  0.0000,  0.0000],
                        [-0.0112,  0.0372, -0.0522],
                        [ 0.0051,  0.0276, -0.2453],
                        [-0.0238, -0.0094, -0.2710],
                        [ 0.0775,  0.0261, -0.0383],
                        [-0.0086, -0.0383, -0.0575],
                        [-0.0031, -0.0275, -0.2436],
                        [-0.0219,  0.0121, -0.2666],
                        [ 0.0827, -0.0221, -0.0394],
                        [-0.0244,  0.0028,  0.0790],
                        [ 0.0170,  0.0028,  0.0876],
                        [ 0.0018, -0.0014,  0.0356],
                        [-0.0212, -0.0085,  0.1343],
                        [ 0.0320,  0.0064,  0.0565],
                        [-0.0120,  0.0455,  0.0724],
                        [-0.0121,  0.0780,  0.0287],
                        [-0.0146,  0.1621, -0.0099],
                        [-0.0047,  0.1687,  0.0081],
                        [ 0.0000,  0.0000,  0.0000],
                        [-0.0150, -0.0527,  0.0714],
                        [-0.0054, -0.0719,  0.0297],
                        [-0.0198, -0.1651, -0.0091],
                        [-0.0038, -0.1708,  0.0043],
                        [ 0.0000,  0.0000,  0.0000]])

        self.topology = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15,
                        16, 17, 11, 19, 20, 21, 22]
                    
        self.connections = []
        for i, pi in enumerate(self.topology):
            if pi >= 0:
                self.connections.append([pi, i])            
                                               
    def forward_kinematics(self, orientation, to_smpl=True, to_mvnx=True):
        positions = super().forward_kinematics(orientation, to_smpl, to_mvnx)
        return positions
