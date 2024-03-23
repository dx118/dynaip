import os
import argparse
import time
import torch
from model.dataset import MotionDataset, collate_fn
from torch.utils.data import DataLoader
from model.model import Poser
from model.loss import loss_vp, loss_p
import random
import utils.config as cfg
import numpy as np

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    random_seed = random.randint(1, 1000)
    random.seed(random_seed)
    np.random.seed(random_seed)    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    print('Seed:', random_seed) 

    parser = argparse.ArgumentParser(description="Training configs")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=180, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default=cfg.save_dir, help="Folder for saving checkpoints and logs")
    parser.add_argument("--save_interval", type=int, default=20, help="Epoch interval for saving model checkpoints")
    parser.add_argument("--train_seg_len", type=int, default=300, help="Maximum length of motion segment")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--datasets", type=list, default=['unipd', 'cip', 'andy', 'emokine', 'virginia'], help="Datasets used for training")

    args = parser.parse_args(args=[])
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading dataset.")
    dataset = MotionDataset(args.datasets, args.train_seg_len, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, collate_fn=collate_fn)

    model = Poser().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs // 2, verbose=False)

    best_loss = float('inf')    
    print(f"Training for {args.num_epochs} Epochs")                        
    for epoch in range(args.num_epochs):
        train_loss = 0.
        pose_loss = 0.
        start_time = time.time()

        for batch_idx, (x, gt_vel, gt_pose, v_init, p_init) in enumerate(dataloader):           
            optimizer.zero_grad()
            v, p = model(x, v_init, p_init)
            loss = loss_vp(v, p, gt_vel, gt_pose)
            _loss_p = loss_p(p, gt_pose)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pose_loss += _loss_p.item()

        end_time = time.time()    
        train_loss /= len(dataloader)
        pose_loss /= len(dataloader)
        scheduler.step()    
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Pose Loss: {pose_loss:.4f}, Epoch Time: {end_time - start_time:.2f}")

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {epoch + 1}.")

        if pose_loss < best_loss:
            best_loss = pose_loss
            if epoch + 1 > 60:
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
                print("Best model saved.")

    print("Finished Training.")


if __name__ == '__main__':
    train()
