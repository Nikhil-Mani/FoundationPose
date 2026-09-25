from Utils import *
import json,os,sys
from pathlib import Path
from torch.utils.data import Dataset
import nvdiffrast.torch as dr
from pathlib import Path
from PIL import Image

event_dir = "/Users/nikhilmani/.cache/huggingface/hub/datasets--mickeykang--Event6D/snapshots/69a0d06a6cc7122da602de81e2cfb8cd4b9478ec/cracker_1101/0000/parsed_events/"

pose_dir =  "/Users/nikhilmani/.cache/huggingface/hub/datasets--mickeykang--Event6D/snapshots/69a0d06a6cc7122da602de81e2cfb8cd4b9478ec/cracker_1101/0000/pose/"

class EventDataReader:
    def __init__(self, data_dir, H=720, W=1280):
        self.H = H
        self.W = W
        self.event_files = sorted(Path(event_dir).glob("*.npz"))

    def get_event_voxel(self, i, num_bins=5):
        path = self.event_files[i]
        ev = np.load(path)['data']
        return events_to_voxel_grid(ev, num_bins=num_bins, H=self.H, W=self.W)  # uses reader's H/W

class EventPoseRefineDataset(Dataset):
    def __init__(
        self,
        event_dir,
        pose_dir,
        calib_dir,
        mesh_path,
        num_bins=5,
        H=720,
        W=1280,
        depth_scale=0.001,
        max_rot_pert_deg=20.0,
        max_trans_pert_m=0.05
    ):
        super().__init__()
        self.event_dir = Path(event_dir)
        self.pose_dir = Path(pose_dir)
        self.calib_dir = Path(calib_dir)
        self.mesh_path = Path(mesh_path)
        self.H = H
        self.W = W
        self.num_bins = num_bins
        self.depth_scale = depth_scale
        self.max_rot_pert_deg = max_rot_pert_deg
        self.max_trans_pert_m = max_trans_pert_m

        # 1. Initialize EventDataReader
        self.event_reader = EventDataReader(self.event_dir, H=H, W=W)

        # 2. Parse Calibration
        calib_yaml_path = self.calib_dir / "calibration.yaml"
        if not calib_yaml_path.exists():
            calib_files = sorted(self.calib_dir.glob("*.yaml"))
            calib_yaml_path = calib_files[0]
        self.cam0_intrinsics, self.cam1_intrinsics, self.T_cam1_cam0 = self.parse_kalibr_yaml(calib_yaml_path)

        # Build Camera Intrinsic Matrix K
        self.K = torch.tensor([
            [self.cam0_intrinsics['fx'], 0, self.cam0_intrinsics['cx']],
            [0, self.cam0_intrinsics['fy'], self.cam0_intrinsics['cy']],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 3. Load Mesh for nvdiffrast
        self.glctx = dr.RasterizeGLContext()
        mesh = trimesh.load(str(self.mesh_path), force='mesh')
        self.vertices = torch.from_numpy(mesh.vertices).float().cuda()
        self.faces = torch.from_numpy(mesh.faces).int().cuda()

        # Pre-compute pixel grid mesh for depth unprojection
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_x = grid_x
        self.grid_y = grid_y

        # Ground truth pose files (.pt or .txt containing 4x4 T_gt matrices)
        self.pose_files = sorted(self.pose_dir.glob("*.pt"))

    def parse_kalibr_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        cam0_raw = calib_data['cam0']['intrinsics']
        cam0_intrinsics = {
            'fx': float(cam0_raw[0]), 'fy': float(cam0_raw[1]),
            'cx': float(cam0_raw[2]), 'cy': float(cam0_raw[3])
        }
        cam1_raw = calib_data['cam1']['intrinsics']
        cam1_intrinsics = {
            'fx': float(cam1_raw[0]), 'fy': float(cam1_raw[1]),
            'cx': float(cam1_raw[2]), 'cy': float(cam1_raw[3])
        }
        T_mat = np.array(calib_data['cam1']['T_cn_cnm1'], dtype=np.float32)
        return cam0_intrinsics, cam1_intrinsics, T_mat

    def render_mesh_xyz(self, T_hypo):
        """Renders 3D XYZ map of CAD mesh at pose hypothesis T_hypo using nvdiffrast."""
        # 1. Transform vertices by T_hypo
        v_cam = (T_hypo[:3, :3] @ self.vertices.T + T_hypo[:3, 3:4]).T  # (V, 3)

        # 2. Project vertices to clip space [-1, 1]
        fx, fy = self.cam0_intrinsics['fx'], self.cam0_intrinsics['fy']
        cx, cy = self.cam0_intrinsics['cx'], self.cam0_intrinsics['cy']

        v_clip_x = (v_cam[:, 0] * fx / v_cam[:, 2] + cx) / (self.W / 2.0) - 1.0
        v_clip_y = 1.0 - (v_cam[:, 1] * fy / v_cam[:, 2] + cy) / (self.H / 2.0)
        v_clip_z = v_cam[:, 2]
        v_clip = torch.stack([v_clip_x, v_clip_y, v_clip_z, torch.ones_like(v_clip_z)], dim=-1).unsqueeze(0)

        # 3. Rasterize
        rast, _ = dr.rasterize(self.glctx, v_clip, self.faces, resolution=[self.H, self.W])

        # Interpolate 3D camera coordinates across rendered face triangles
        xyz_rendered, _ = dr.interpolate(v_cam.unsqueeze(0), rast, self.faces)
        xyz_rendered = xyz_rendered.squeeze(0).permute(2, 0, 1)  # (3, H, W)

        # Mask background pixels where rasterizer depth is zero
        mask = (rast[0, :, :, 3] > 0).float().unsqueeze(0)
        xyz_rendered = xyz_rendered * mask

        return xyz_rendered.cpu()

    def generate_random_pose_perturbation(self, T_gt):
        """Adds uniform noise to T_gt to simulate a pose hypothesis T_hypo."""
        # Random rotation matrix
        rand_axis = torch.randn(3)
        rand_axis = rand_axis / torch.norm(rand_axis)
        rand_angle = torch.rand(1) * np.radians(self.max_rot_pert_deg)

        # Rodrigues rotation matrix formula
        K_mat = torch.tensor([
            [0, -rand_axis[2], rand_axis[1]],
            [rand_axis[2], 0, -rand_axis[0]],
            [-rand_axis[1], rand_axis[0], 0]
        ])
        R_pert = torch.eye(3) + torch.sin(rand_angle) * K_mat + (1 - torch.cos(rand_angle)) * (K_mat @ K_mat)

        # Random translation noise (meters)
        t_pert = (torch.rand(3) - 0.5) * 2.0 * self.max_trans_pert_m

        # Form T_delta
        T_delta = torch.eye(4)
        T_delta[:3, :3] = R_pert
        T_delta[:3, 3] = t_pert

        # Pose Hypothesis T_hypo = T_gt @ T_delta
        T_hypo = T_gt @ T_delta

        # Target relative transformation for network to predict: T_delta_gt = T_gt @ inv(T_hypo)
        gt_delta_rot = T_delta[:3, :3].T
        gt_delta_trans = -gt_delta_rot @ T_delta[:3, 3]

        return T_hypo, gt_delta_rot, gt_delta_trans

    def load_metric_depth_xyz(self, frame_idx):
        depth_path = self.event_dir / "depth" / f"{frame_idx:06d}.png"
        depth_raw = np.array(Image.open(depth_path), dtype=np.uint16)
        depth_m = torch.from_numpy(depth_raw.astype(np.float32)) * self.depth_scale

        fx, fy = self.cam0_intrinsics['fx'], self.cam0_intrinsics['fy']
        cx, cy = self.cam0_intrinsics['cx'], self.cam0_intrinsics['cy']

        x_map = (self.grid_x - cx) * depth_m / fx
        y_map = (self.grid_y - cy) * depth_m / fy
        xyz_map = torch.stack([x_map, y_map, depth_m], dim=0)

        return xyz_map

    def __len__(self):
        return len(self.event_reader.event_files)

    def __getitem__(self, idx):
        # 1. Load Ground Truth Pose T_gt from pose_dir
        T_gt = torch.load(self.pose_files[idx])  # Shape: (4, 4)

        # 2. Perturb T_gt to make a synthetic candidate pose hypothesis T_hypo
        T_hypo, gt_delta_rot, gt_delta_trans = self.generate_random_pose_perturbation(T_gt)

        # --- INPUT A: Real Observed Event Voxel + Real Depth XYZ ---
        event_voxel = self.event_reader.get_event_voxel(idx, num_bins=self.num_bins)
        event_tensor_A = torch.from_numpy(event_voxel).float()
        xyz_A = self.load_metric_depth_xyz(idx)
        input_A = torch.cat([event_tensor_A, xyz_A], dim=0)

        # --- INPUT B: Zero Dummy Events + Rendered CAD XYZ at T_hypo ---
        event_tensor_B = torch.zeros((self.num_bins, self.H, self.W), dtype=torch.float32)
        xyz_B = self.render_mesh_xyz(T_hypo.cuda())
        input_B = torch.cat([event_tensor_B, xyz_B], dim=0)

        return {
            'input_A': input_A,
            'input_B': input_B,
            'gt_delta_rot': gt_delta_rot,
            'gt_delta_trans': gt_delta_trans
        }
