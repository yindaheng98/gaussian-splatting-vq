import torch
import numpy as np


def spatial_split(xyz: torch.Tensor, xyz_tile_size):
    tile_size = torch.Tensor(xyz_tile_size[:3]).to(xyz.device)
    floors = (xyz/tile_size.unsqueeze(0)).floor()
    blkids = floors.type(torch.int32)
    print("spatial_split", xyz_tile_size, "blks", blkids.unique(dim=0).shape[0], blkids.max(dim=0).values-blkids.min(dim=0).values)
    rests = xyz/tile_size - floors
    return blkids, rests


def split_by_blkids(blkids: torch.Tensor, *tensors: torch.Tensor):
    for tensor in tensors:
        assert tensor.shape[0] == tensors[0].shape[0]
    distinct_blkids = blkids.unique(dim=0)
    tensor_blks = [[None] * distinct_blkids.shape[0] for _ in range(len(tensors))]
    item_idx = [None] * distinct_blkids.shape[0]
    idx = torch.from_numpy(np.asarray(range(blkids.shape[0]))).to(blkids.device)
    for i in range(distinct_blkids.shape[0]):
        inblk_idx = (blkids == distinct_blkids[i:i+1, ...]).all(dim=1)
        item_idx[i] = idx[inblk_idx]
        for j, tensor in enumerate(tensors):
            tensor_blks[j][i] = tensor[inblk_idx, ...]
    return distinct_blkids, item_idx, *tensor_blks


def spatial_merge(blkids: torch.Tensor, rests: torch.Tensor, xyz_tile_size):
    print("spatial_merge", xyz_tile_size, "blks", "blks", blkids.max(dim=0).values-blkids.min(dim=0).values)
    tile_size = torch.Tensor(xyz_tile_size[:3]).to(rests.device)
    floors = blkids.type(torch.float32)
    data = (floors+rests)*tile_size.unsqueeze(0)
    return data


def spatial_split_octree(xyz: torch.Tensor, xyz_tile_size_base=[1, 1, 1], xyz_tile_size_maxlevel=4, split_thr=5e3):
    tile_size_base = torch.from_numpy(np.asarray(xyz_tile_size_base[:3])).to(torch.int32).to(xyz.device)
    octree_blkids = torch.IntTensor(size=xyz.shape).to(device=xyz.device)
    octree_blksizes = torch.IntTensor(size=xyz.shape).to(device=xyz.device)
    octree_rests = torch.zeros_like(xyz)
    octree_blk_istbd = torch.BoolTensor(size=(xyz.shape[0],)).to(device=xyz.device)
    octree_blkids[...] = 0
    octree_blk_istbd[...] = True
    for level in reversed(range(xyz_tile_size_maxlevel+1)):
        tile_size = tile_size_base * 2**level
        xyz_tbd = xyz[octree_blk_istbd, ...]
        blkids, rests = spatial_split(xyz_tbd, tile_size)
        distinct_blkids, _, blks = split_by_blkids(blkids, rests)
        blk_counts = torch.IntTensor([blk.shape[0] for blk in blks]).to(rests.device)
        octree_distinct_blkids = distinct_blkids[blk_counts < split_thr, ...] if level > 0 else distinct_blkids
        octree_rests_tbd = octree_rests[octree_blk_istbd, ...]
        octree_blkids_tbd = octree_blkids[octree_blk_istbd, ...]
        octree_blksizes_tbd = octree_blksizes[octree_blk_istbd, ...]
        octree_blk_istbd_ = octree_blk_istbd[octree_blk_istbd]
        for i in range(octree_distinct_blkids.shape[0]):
            inblk_idx = (blkids == octree_distinct_blkids[i:i+1, ...]).all(dim=1)
            octree_blkids_tbd[inblk_idx] = blkids[inblk_idx]
            octree_blksizes_tbd[inblk_idx, ...] = tile_size
            octree_rests_tbd[inblk_idx] = rests[inblk_idx]
            octree_blk_istbd_[inblk_idx] = False
        octree_blkids[octree_blk_istbd, ...] = octree_blkids_tbd
        octree_blksizes[octree_blk_istbd, ...] = octree_blksizes_tbd
        octree_rests[octree_blk_istbd, ...] = octree_rests_tbd
        octree_blk_istbd[octree_blk_istbd.clone()] = octree_blk_istbd_
    return octree_blkids, octree_blksizes, octree_rests
