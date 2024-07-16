import torch


def spatial_split(xyz: torch.Tensor, xyz_tile_size):
    tile_size = torch.FloatTensor(xyz_tile_size[:3]).to(xyz.device)
    floors = (xyz/tile_size.unsqueeze(0)).floor()
    blkids = floors.type(torch.int32)
    print("spatial_split", xyz_tile_size, "blks", blkids.unique(dim=0).shape[0], blkids.max(dim=0).values-blkids.min(dim=0).values)
    rests = xyz - floors*tile_size
    return blkids, rests


def split_by_blkids(blkids: torch.Tensor, *tensors: torch.Tensor):
    for tensor in tensors:
        assert tensor.shape[0] == tensors[0].shape[0]
    distinct_blkids = blkids.unique(dim=0)
    tensor_blks = [[None for _ in range(distinct_blkids.shape[0])] for _ in range(len(tensors))]
    for i in range(distinct_blkids.shape[0]):
        inblk_idx = (blkids == distinct_blkids[i:i+1, ...]).all(dim=1)
        for j, tensor in enumerate(tensors):
            tensor_blks[j][i] = tensor[inblk_idx, ...]
    return distinct_blkids, *tensor_blks


def spatial_merge(blkids: torch.Tensor, rests: torch.Tensor, xyz_tile_size):
    print("spatial_merge", xyz_tile_size, "blks", "blks", blkids.max(dim=0).values-blkids.min(dim=0).values)
    tile_size = torch.FloatTensor(xyz_tile_size[:3]).to(rests.device)
    floors = blkids.type(torch.float32)
    data = floors*tile_size.unsqueeze(0)+rests
    return data
