import numpy as np
import torch
import matplotlib.pyplot as plt
from lib.utils_nca import make_sphere_mask

class VoxelGeo:

    def __init__(self, size: int):
        self.size = size
        self._size = size
        self.voxels = torch.zeros((size, size, size, 4), dtype=torch.float32)
        self.history = []

    def update(self, voxels, history=False):
        if history: self.history.append(self.voxels)
        self.voxels = voxels

    def pad(self, p):
        if p <= 0: return
        voxels = torch.zeros((self.size + p * 2, self.size + p * 2, self.size + p * 2, 4), dtype=torch.float32)
        voxels[p:p+self.size, p:p+self.size, p:p+self.size, :] = self.voxels
        self.voxels = voxels
        self.size += p * 2

    def premult_alpha(self):
        self.voxels[..., :-1] *= self.voxels[..., -1].unsqueeze(-1)

    def cowlor(self, n: int, radius_min=1, radius_max=1):
        assert 0.0 < radius_min <= radius_max
        coords = torch.nonzero(self.voxels)
        col_mask = torch.zeros_like(self.voxels)
        for i in range(n):
            radius = np.random.rand() * (radius_max - radius_min) + radius_min
            # center = torch.randint(0, self.size, (3,))
            center = coords[np.random.randint(0, coords.shape[0]), :-1]
            col = torch.rand((1, 1, 1, 3), dtype=torch.float32)
            mask = 1.0 - make_sphere_mask(self.size, center, radius).repeat(1, 1, 1, 4)
            mask[..., :-1] *= col
            col_mask = col_mask - (col_mask[..., -1] * mask[..., -1]).unsqueeze(-1) + mask
        col_mask = torch.clamp(col_mask, 0, 1)
        col_mask[col_mask == 0] = 1
        self.voxels[..., :-1] *= col_mask[..., :-1]
        self.premult_alpha()

    @staticmethod
    def _plot_fig(voxels, elev, azim, show_edges=True):
        edge_color = np.zeros((3 if show_edges else 4), dtype=np.float32)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        voxels_t = voxels.transpose(1, 2)
        ax.voxels(voxels_t[:, :, :, -1].numpy(), facecolors=voxels_t[:, :, :, :3].numpy(), edgecolor=edge_color)
        return fig, ax

    def plot(self, return_array=False, filename=None, elev=35, azim=45, show_edges=True):
        fig, ax = VoxelGeo._plot_fig(self.voxels, elev, azim, show_edges)
        if return_array:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return data
        elif filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def load_file(filename):
        import json
        with open(filename) as f:
            model_data = json.load(f)
        dim = model_data["dimension"][0]
        dim_x, dim_y, dim_z = int(dim["width"]), int(dim["height"]), int(dim["depth"])
        dim_max = max([dim_x, dim_y, dim_z])
        model = VoxelGeo(dim_max + 1)
        voxels = [ (int(voxel["x"]), int(voxel["y"]), int(voxel["z"])) for voxel in model_data["voxels"] ]
        voxels = torch.tensor(voxels, dtype=torch.int64)
        voxels[:, 0] += (dim_max - dim_x) // 2
        voxels[:, 1] += (dim_max - dim_y) // 2
        voxels[:, 2] += (dim_max - dim_z) // 2
        model.voxels[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
        return model