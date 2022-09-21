from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire

import torch

from ddpm_tutorial.unet import Unet
from ddpm_tutorial.dataset_toyzero import Toyzero
from ddpm_tutorial.ddpm import DDPM


class DDPMTranslate():
    """
    DDPM model training and inference
    """
    def __init__(self,
                 encoder_ckpt,
                 decoder_ckpt,
                 timesteps=1000,
                 encode_step=200,
                 decode_step=200):

        # set up models
        self.encoder = DDPM(
            Unet(dim=32, dim_mults=(1, 2, 4,), channels=1),
            timesteps=timesteps,
            img_sz=(1, 256, 256)
        )
        self.encoder.load_state_dict(torch.load(encoder_ckpt))
        self.encoder.cuda()
        
        self.decoder = DDPM(
            Unet(dim=32, dim_mults=(1, 2, 4,), channels=1),
            timesteps=timesteps,
            img_sz=(1, 256, 256)
        )
        self.decoder.load_state_dict(torch.load(decoder_ckpt))
        self.decoder.cuda()

        self.encode_step = encode_step
        self.decode_step = decode_step

    def translate(self, tensor, record_freq=None):
        """
        Use saved model at epoch
        """
        code = self.encoder.encode(tensor, self.encode_step)
        if record_freq is None:
            record_freq = self.decode_step
        return self.decoder.decode(code, self.decode_step, record_freq)



# if __name__ == "__main__":
#     fire.Fire(DDPMTranslate)
