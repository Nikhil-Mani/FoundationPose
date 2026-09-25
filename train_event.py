import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import your Dataset class (incorporating nvdiffrast rendering)
from dataset import EventPoseRefineDataset

# --- Loss Function for Pose Refinement ---
class PoseRefineLoss(nn.Module):
    def __init__(self, trans_weight=1.0, rot_weight=1.0):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight

    def forward(self, pred_rot, pred_trans, gt_rot, gt_trans):
        """
        pred_rot: (B, 3, 3) or (B, 6) rotation representation
        pred_trans: (B, 3) translation vector
        gt_rot: (B, 3, 3) ground truth relative rotation
        gt_trans: (B, 3) ground truth relative translation
        """
        # Translation L1 / Smooth L1 loss
        loss_trans = F.smooth_l1_loss(pred_trans, gt_trans)

        # Rotation geodesic loss or Frobenius norm loss: ||R_pred - R_gt||_F
        loss_rot = F.mse_loss(pred_rot, gt_rot)

        total_loss = self.trans_weight * loss_trans + self.rot_weight * loss_rot
        return total_loss, loss_trans, loss_rot


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_trans_loss = 0.0
    total_rot_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_A = batch['input_A'].to(device, non_blocking=True)  # (B, num_bins + 3, H, W)
        input_B = batch['input_B'].to(device, non_blocking=True)  # (B, num_bins + 3, H, W)
        gt_delta_rot = batch['gt_delta_rot'].to(device, non_blocking=True)  # (B, 3, 3)
        gt_delta_trans = batch['gt_delta_trans'].to(device, non_blocking=True)  # (B, 3)

        optimizer.zero_grad()

        # Forward pass: predicts predicted delta rotation and translation
        pred_rot, pred_trans = model(input_A, input_B)

        loss, l_trans, l_rot = criterion(pred_rot, pred_trans, gt_delta_rot, gt_delta_trans)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_trans_loss += l_trans.item()
        total_rot_loss += l_rot.item()

        if step % 10 == 0:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Trans_Loss", l_trans.item(), global_step)
            writer.add_scalar("Train/Rot_Loss", l_rot.item(), global_step)
            print(
                f"Epoch [{epoch}][{step}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (Trans: {l_trans.item():.4f}, Rot: {l_rot.item():.4f})"
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    for batch in dataloader:
        input_A = batch['input_A'].to(device, non_blocking=True)
        input_B = batch['input_B'].to(device, non_blocking=True)
        gt_delta_rot = batch['gt_delta_rot'].to(device, non_blocking=True)
        gt_delta_trans = batch['gt_delta_trans'].to(device, non_blocking=True)

        pred_rot, pred_trans = model(input_A, input_B)
        loss, _, _ = criterion(pred_rot, pred_trans, gt_delta_rot, gt_delta_trans)
        val_loss += loss.item()

    return val_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Event-based 6D Pose Refinement Network")
    parser.add_argument("--event_dir", type=str, required=True, help="Path to event data directory")
    parser.add_argument("--pose_dir", type=str, required=True, help="Path to ground truth pose directory")
    parser.add_argument("--calib_dir", type=str, required=True, help="Path to calibration YAML directory")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to CAD mesh file (.obj/.ply)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # 1. Dataset & DataLoader (Uses T_gt directly to generate perturbed hypotheses)
    train_dataset = EventPoseRefineDataset(
        event_dir=args.event_dir,
        pose_dir=args.pose_dir,
        calib_dir=args.calib_dir,
        mesh_path=args.mesh_path,
        max_rot_pert_deg=20.0,
        max_trans_pert_m=0.05
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 2. Instantiate Model (Replace with your actual RefineNet architecture)
    # model = RefineNet(in_channels=5+3).to(device)
    model = RefineNet(in_channels=8).to(device)

    # 3. Optimizer, Scheduler, and Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = PoseRefineLoss(trans_weight=1.0, rot_weight=1.0)

    best_loss = float("inf")

    # 4. Training Loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        scheduler.step()

        print(f"--- Epoch {epoch} Complete | Avg Train Loss: {train_loss:.4f} ---")

        # Save Checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }

        # Save last
        torch.save(checkpoint, os.path.join(args.output_dir, "refine_net_last.pt"))

        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "refine_net_best.pt"))
            print(f"New best checkpoint saved at epoch {epoch}")

    writer.close()


if __name__ == "__main__":
    main()
