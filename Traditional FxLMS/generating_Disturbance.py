import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import repeat 
from scipy import signal, misc
import torch

#-------------------------------------------------------------
def Disturbance_generation_from_real_noise(fs, Repet, wave_from, Pri_path, Sec_path):
    wave  = wave_from
    wavec = wave
    for ii in range(Repet):
        wavec = np.concatenate((wavec,wave),axis=0)
    pass
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, wavec), signal.lfilter(Sec_path, 1, wavec)
    
    N   = len(Dir)
    N_z = N//fs 
    Dir, Fx = Dir[0:N_z*fs], Fx[0:N_z*fs]
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Fx).type(torch.float), torch.from_numpy(wavec).type(torch.float)

