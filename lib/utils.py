import torch
import os

def print_tensor(x):
    print(x)
    if type(x) == torch.Tensor:
        print("\t%s    %s    %s\n" % (str(x.shape), str(x.dtype), str(x.device)))

def make_video():
    cmd = "\"ffmpeg-4.4.1-essentials_build\\bin\\ffmpeg.exe\" -f image2 -r 5 -y " \
          "-i _results\\tpose_regenerating\\tpose_regenerating_frame%d.jpg test.mp4"
    os.system(cmd)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp