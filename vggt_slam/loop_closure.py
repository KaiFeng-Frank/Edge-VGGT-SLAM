import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import heapq
from typing import NamedTuple
import torchvision.transforms as T
import os
import sys
from pathlib import Path

from vggt_slam.runtime import resolve_device

SALAD_CHECKPOINT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


class _DinoBackbone(torch.nn.Module):
    """The SALAD DINOv2 backbone, without a second pretrained weight download."""

    def __init__(self, offline=False):
        super().__init__()
        local_repo = Path(__file__).resolve().parents[1] / "third_party" / "dinov2"
        if not (local_repo / "hubconf.py").is_file():
            cached_repos = sorted(Path(torch.hub.get_dir()).glob("facebookresearch_dinov2_*"))
            local_repo = next((repo for repo in cached_repos if (repo / "hubconf.py").is_file()), None)
        if local_repo is not None and (local_repo / "hubconf.py").is_file():
            self.model = torch.hub.load(str(local_repo), "dinov2_vitb14", source="local", pretrained=False)
        elif offline:
            raise FileNotFoundError(
                "DINOv2 source is unavailable offline. Copy facebookresearch/dinov2 "
                "into third_party/dinov2, or disable loop closures with --max_loops 0."
            )
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=False,
                trust_repo=True, skip_validation=True,
            )

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)
        token = x[:, 0]
        features = x[:, 1:].reshape(batch_size, height // 14, width // 14, 768)
        return features.permute(0, 3, 1, 2), token


class _SaladInference(torch.nn.Module):
    """Same pretrained backbone/aggregator as SALAD's Lightning training class."""

    def __init__(self, offline=False):
        super().__init__()
        local_repo = Path(__file__).resolve().parents[1] / "third_party" / "salad"
        if (local_repo / "salad").is_dir() and str(local_repo) not in sys.path:
            sys.path.insert(0, str(local_repo))
        from salad.models_salad.aggregators.salad import SALAD

        self.backbone = _DinoBackbone(offline=offline)
        self.aggregator = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)

    def forward(self, images):
        return self.aggregator(self.backbone(images))

tensor_transform = T.ToPILImage()
denormalize = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)

class LoopMatch(NamedTuple):
    similarity_score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int

class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.heap = []  # Simulated max-heap by negating scores

    def add(self, match: LoopMatch):
        # Negate similarity_score to turn min-heap into max-heap
        item = (-match.similarity_score, match)
        # item = (-match.detected_submap_id, match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            # Push new element and remove the largest (i.e., smallest negated)
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        """Return sorted list of matches (lowest value first)"""
        return [match for _, match in sorted(self.heap, reverse=True)]
        

class ImageRetrieval:
    def __init__(self, input_size=224, device="auto", checkpoint=None, offline=False, batch_size=1):
        self.device = resolve_device(device)
        ckpt_pth = Path(checkpoint).expanduser() if checkpoint is not None else Path(torch.hub.get_dir()) / "checkpoints/dino_salad.ckpt"
        if checkpoint is None and not ckpt_pth.is_file() and not offline:
            ckpt_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(SALAD_CHECKPOINT_URL, str(ckpt_pth))
        if not ckpt_pth.is_file():
            raise FileNotFoundError(f"SALAD checkpoint not found: {ckpt_pth}. Download {SALAD_CHECKPOINT_URL} or use --max_loops 0.")
        self.model = _SaladInference(offline=offline)
        state_dict = torch.load(ckpt_pth, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=torch.float32)
        self.batch_size = max(1, int(batch_size))
        self.transform = input_transform((input_size, input_size))

    def get_single_embeding(self, cv_img):
        with torch.no_grad():
            pil_img = self.transform(tensor_transform(cv_img))
            return self.model(pil_img.unsqueeze(0).to(self.device))

    def get_batch_descriptors(self, imgs):
        # Expecting imgs to be a batch of images (B, C, H, W)
        with torch.no_grad():
            pil_imgs = [tensor_transform(img) for img in imgs]  # Convert each tensor to PIL Image
            descriptors = []
            for start in range(0, len(pil_imgs), self.batch_size):
                batch = torch.stack([self.transform(img) for img in pil_imgs[start:start + self.batch_size]])
                descriptors.append(self.model(batch.to(self.device)))
            return torch.cat(descriptors, dim=0)
    
    def get_all_submap_embeddings(self, submap):
        # Frames is np array of shape (S, 3, H, W)
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(self, map, submap, max_similarity_thres = 0.80, max_loop_closures = 0):
        if max_loop_closures <= 0:
            return []
        matches_queue = LoopMatchQueue(max_size=max_loop_closures)
        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(query_vector, submap.get_id(), ignore_last_submap=True)
            if best_score < max_similarity_thres:
                new_match_data = LoopMatch(best_score, submap.get_id(), query_id, best_submap_id, best_frame_id)
                matches_queue.add(new_match_data)
            query_id += 1
        
        return matches_queue.get_matches()
