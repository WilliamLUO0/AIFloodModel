import torch, sys
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("python:", sys.executable)

import os
from os import path as osp

arch_folder = osp.dirname(osp.abspath(__file__))

print(osp.abspath(__file__))
print(arch_folder)
print((os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
print(root_path)
print(osp.join(root_path, 'tb_logger', "001_PFT_SRx2_scratch"))