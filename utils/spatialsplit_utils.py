import torch


def spatial_split(xyz: torch.Tensor, xyz_tile_size):
    tile_size = torch.FloatTensor([xyz_tile_size[0], xyz_tile_size[1], xyz_tile_size[2]]).to(xyz.device)
    floors = (xyz/tile_size.unsqueeze(0)).floor()
    blkids = floors.type(torch.int32)
    print("spatial_split", xyz_tile_size, "blks", blkids.unique(dim=0).shape[0], blkids.max(dim=0).values-blkids.min(dim=0).values)
    rests = xyz - floors*tile_size
    return blkids, rests


def spatial_merge(blkids: torch.Tensor, rests: torch.Tensor, xyz_tile_size):
    print("spatial_merge", xyz_tile_size, "blks", "blks", blkids.max(dim=0).values-blkids.min(dim=0).values)
    tile_size = torch.FloatTensor([xyz_tile_size[0], xyz_tile_size[1], xyz_tile_size[2]]).to(rests.device)
    floors = blkids.type(torch.float32)
    data = floors*tile_size.unsqueeze(0)+rests
    return data
