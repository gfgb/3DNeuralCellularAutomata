import torch
import os
import imageio
from tqdm import tqdm
import sys

from matplotlib import MatplotlibDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from lib.voxel_geo import VoxelGeo
from lib.nca_model import  NCAModel3D, NCAModel3D_conv
from lib.utils_nca import make_seed_3d, make_sphere_mask
from lib.params import Params
from lib.options import parse_rendering_options


if __name__ == "__main__":

    # options = "--experiment-name camel_conv_regenerating_e5000_h300 " \
    #           "--steps 200 " \
    #           "--fps 15 " \
    #           "--azim 0 " \
    #           "--rotation-speed 1 " \
    #           "-dmg 10 2 10 7 50 " \
    #           "-dmg 10 16 15 6 110"

    if len(sys.argv) > 1: options = None

    args, args_str = parse_rendering_options(options_str=options)
    print(args_str)

    damage = dict()
    if args.damage is not None:
        damage = { d[-1]:{"center": d[:-2], "radius": d[-2]} for d in args.damage }

    MODEL_PATH = os.path.join("models", args.experiment_name + ".pth")
    PARAMS_PATH = os.path.join("models", args.experiment_name + "_params.json")
    LOSS_LOG_PATH = os.path.join("models", args.experiment_name + "_loss.txt")

    out_video_filename = "_results/%s.mp4" % args.experiment_name

    if os.path.isfile(out_video_filename):
        print("Error: a file with the same name already exists")
        exit()

    writer = imageio.get_writer(out_video_filename, fps=args.fps)

    device = torch.device("cpu")
    params = Params.load(PARAMS_PATH)

    nca_model = NCAModel3D(params.n_channels, params.cell_fire_rate, device, hidden_size=params.hidden_sz)
    # nca_model = NCAModel3D_conv(params.n_channels, params.cell_fire_rate, device, hidden_size=params.hidden_sz)
    nca_model.load_state_dict(torch.load(MODEL_PATH))
    nca_model = nca_model.eval().to(device)

    output = make_seed_3d(params.voxel_grid_size, params.n_channels).unsqueeze(0).to(device)

    vm = VoxelGeo(params.voxel_grid_size)

    with torch.no_grad():

        for i in tqdm(range(args.steps)):

            output = nca_model(output, steps=1)
            voxels = output[0, :, :, :, :4].clone().detach().cpu()
            voxels = torch.clamp(voxels, 0, 1)
            keep = voxels[..., -1] > 0.8
            voxels *= voxels * keep.unsqueeze(-1)

            if i in damage:
                dmg = damage[i]
                mask = make_sphere_mask(params.voxel_grid_size,
                                        torch.tensor(dmg["center"], dtype=torch.float32),
                                        dmg["radius"])
                voxels *= mask
                output[0, :, :, :, :4] *= mask

            vm.update(voxels)
            img = vm.plot(return_array=True, azim=args.azim + i * args.rotation_speed)
            writer.append_data(img)

    writer.close()