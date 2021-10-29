import argparse
import pprint

def parse_training_options(options_str=None):

    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group("global")
    global_group.add_argument("--name", type=str,help="Name of the experiment")
    global_group.add_argument("--file", type=str, help="Name of the data file to use")
    global_group.add_argument("-exp", "--experiment-type", type=str, default="regenerating", help="Experiment type")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--lr", type=float, default=2e-3, help="Training learning rate")
    train_group.add_argument("--lr-decay", type=float, default=0.9999, help="Training learning rate decay")
    train_group.add_argument("--betas", nargs=2, type=float, default=(0.5, 0.5), help="Coefficients for Adam")
    train_group.add_argument("--epochs", type=int, default=8000, help="Number of training epochs")
    train_group.add_argument("--batch-sz", type=int, default=8, help="Batch size")
    train_group.add_argument("--pool-sz", type=int, default=1024, help="Pool size")
    train_group.add_argument("--steps", nargs=2, type=int, default=(8, 32), help="Min and max steps for each training "
                                                                                 "epoch")

    net_group = parser.add_argument_group("network")
    net_group.add_argument("--trainable-conv", type=int, default=0, help="Use trainable convolutional layer")
    net_group.add_argument("--hidden-sz", type=int, default=256, help="Size of the hidden layer of the NN")
    net_group.add_argument("--n-channels", type=int, default=16, help="Number of auxiliary channels for the NN")
    net_group.add_argument("--cell-fire-rate", type=float, default=0.5, help="Random dropout of cells from updating "
                                                                             "to simulate asynchronicity")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--padding", type=int, default=0, help="Padding applied to the voxelized data")

    return argparse_to_str(parser, options_str)

def parse_rendering_options(options_str=None):

    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group("global")
    global_group.add_argument("--experiment-name", type=str,help="Name of the experiment and name of the output video")
    global_group.add_argument("--steps", type=int, help="Number of updates to perform")
    global_group.add_argument("--fps", type=int, default=15, help="Framerate")
    global_group.add_argument("--azim", type=int, default=0, help="Starting position of the camera wrt the center")
    global_group.add_argument("--rotation-speed", type=int, default=1, help="Rotation of the camera")
    global_group.add_argument("-dmg", "--damage", nargs=5, type=int, action="append",
                              help="A tuple (x, y, z, r, f) where x, y and z are the coordinates of the center of a "
                                   "sphere; r is its radius; f is the time (as number of updates) when you want the "
                                   "damage to be applied The sphere is used to damage the volume "
                                   "(i.e. set areas of the volume to zero)")

    return argparse_to_str(parser, options_str)

def argparse_to_str(parser, options_str=None):

    if options_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options_str.split())

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    # args_str = f'```{args_str}```'

    return args, args_str
