import sys

import torch
import torch.optim as optim

import os

from matplotlib import MatplotlibDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

from lib.nca_model import NCAModel3D, NCAModel3D_conv
from lib.utils_nca import make_seed_3d
from lib.utils import print_tensor as printt
from lib.trainer import Trainer
from lib.voxel_geo import VoxelGeo
from lib.options import parse_training_options
from lib.params import Params

if __name__ == "__main__":

    # options = "--name camel " \
    #           "--trainable-conv 0 " \
    #           "--file camel.json " \
    #           "--experiment-type regenerating " \
    #           "--epochs 5000 " \
    #           "--hidden-sz 3000 "

    # options = "--name camel_conv " \
    #           "--trainable-conv 1 " \
    #           "--file camel.json " \
    #           "--experiment-type regenerating " \
    #           "--epochs 5000 " \
    #           "--hidden-sz 128 "
    if len(sys.argv) > 1: options = None

    args, args_str = parse_training_options(options_str=options)
    print(args_str)
    params = Params(vars(args))

    exp_name = "%s_%s_e%d_h%d" % (params.name, params.experiment_type, params.epochs, params.hidden_sz)

    MODEL_PATH = os.path.join("models", exp_name + ".pth")
    PARAMS_PATH = os.path.join("models", exp_name + "_params.json")
    LOSS_LOG_PATH = os.path.join("models", exp_name + "_loss.txt")
    VOXGEO_PATH = os.path.join("data", params.file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_geo = VoxelGeo.load_file(VOXGEO_PATH)
    target_geo.pad(params.padding)
    params.add("voxel_grid_size", target_geo.size)

    target_geo.cowlor(n=10, radius_min=2, radius_max=6)
    target_geo.plot()
    target_geo.plot(filename=os.path.join("models", exp_name + "_target.png"))

    target = target_geo.voxels
    target = target.unsqueeze(0).repeat(params.batch_sz, 1, 1, 1, 1)

    seed = make_seed_3d(params.voxel_grid_size, params.n_channels)

    ####################################################################################

    nca_model = NCAModel3D if not params.trainable_conv else NCAModel3D_conv

    nca_model = nca_model(n_channels=params.n_channels,
                           fire_rate=params.cell_fire_rate,
                           device=device,
                           hidden_size=params.hidden_sz)

    if params.use_pretrained: nca_model.load_pretrained(MODEL_PATH)

    optimizer = optim.Adam(nca_model.parameters(), lr=params.lr, betas=params.betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, params.lr_decay)
    loss_function = torch.nn.MSELoss()

    trainer = Trainer(params=params,
                      model=nca_model,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      scheduler=scheduler)

    trainer.train(device=device,
                  seed=seed,
                  target=target,
                  epochs=params.epochs,
                  model_path=MODEL_PATH,
                  hparams_path=PARAMS_PATH,
                  loss_log_path=LOSS_LOG_PATH)