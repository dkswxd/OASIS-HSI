

import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.sfid_scores import sfid_pytorch
import config
import numpy as np


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create utils ---#
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
fid_computer = sfid_pytorch(opt, dataloader_val)

ans = fid_computer.compute_fid_file('/home/qiu/zzh/Swin-Spectral/data/HSI/crn_dir')
print(ans)
ans = np.array(ans)
print(ans.mean())


