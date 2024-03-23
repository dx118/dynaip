import torch.nn.functional as F

def loss_vp(v, p, gt_vel, gt_pose):
    loss1 = 0.
    for pred, target in zip(v, gt_vel):
        # match the subposer velocity order
        loss = F.mse_loss(pred, target[:, [2, 3, 0, 4, 5, 1, 0, 1]].flatten(1))
        loss1 += loss
    loss1 = loss1 / len(gt_pose) 
    
    loss2 = 0.
    for pred, target in zip(p, gt_pose):
        # match the subposer pose order
        loss = F.mse_loss(pred, target[:, [7, 9, 8, 10, 0, 1, 2, 3, 4, 5, 6]].flatten(1))
        loss2 += loss
    loss2 = loss2 / len(gt_pose)
        
    total_loss = loss1 + loss2
    return total_loss


def loss_p(p, gt_pose):
    loss2 = 0.
    for pred, target in zip(p, gt_pose):
        loss = F.mse_loss(pred, target[:, [7, 9, 8, 10, 0, 1, 2, 3, 4, 5, 6]].flatten(1))
        loss2 += loss
    loss2 = loss2 / len(gt_pose)
    return loss2