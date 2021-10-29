import torch

def make_seed_3d(sz, n_channels, pos=None):
    if pos is None: pos = (sz // 2, sz // 2, sz // 2)
    seed = torch.zeros([sz, sz, sz, n_channels], dtype=torch.float32)
    seed[pos[0], pos[1], pos[2], 3:] = 1.0
    return seed

def make_sphere_mask(sz, center, radius):
    coords = torch.zeros((sz ** 3, 3), dtype=torch.float32)
    coords[:, 0] = torch.arange(0, sz).repeat_interleave(sz ** 2)
    coords[:, 1] = torch.arange(0, sz).repeat_interleave(sz).repeat(sz)
    coords[:, 2] = torch.arange(0, sz).repeat(sz ** 2)
    dist = torch.linalg.norm(coords - center, ord=2, dim=1) <= radius
    coords = coords[dist, :].long()
    mask = torch.ones((sz, sz, sz, 1), dtype=torch.float32)
    mask[coords[:, 0], coords[:, 1], coords[:, 2], :] = 0
    return mask

def make_rand_sphere_masks(n, sz, radius=None):
    if n == 0: return []
    masks = []
    for _ in range(n):
        center = torch.randint(0, sz, (3,))
        radius = torch.randint(0, sz // 2, (1,)) if radius is None else radius
        masks.append(make_sphere_mask(sz, center, radius))
    return torch.stack(masks, dim=0)