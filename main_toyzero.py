from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire

import torch

from ddpm_tutorial.unet import Unet
from ddpm_tutorial.dataset_toyzero import Toyzero
from ddpm_tutorial.ddpm import DDPM

DATA_DIR = Path("./data/")
CKPT_DIR = Path("./checkpoints/")
IMG_DIR = Path("./images/")
TOYZERO_DIR = Path('/data/datasets/LS4GAN/toy-adc_256x256_precropped')


class DDPMRun():
    """
    DDPM model training and inference
    """
    def __init__(self, timesteps=1000, batch_size=16, run_name=None):

        self.batch_size = batch_size
        self.timesteps = timesteps

        # set up checkpoint folder
        if run_name:
            self.ckpt_dir = CKPT_DIR/run_name
        else:
            self.ckpt_dir = CKPT_DIR
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        print(f'Checkpoints will be saved to or loaded from{self.ckpt_dir}')

        # set up model
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=1)
        self.ddpm = DDPM(model, timesteps=self.timesteps, img_sz=(1, 256, 256))

        self.run_name = run_name


    def train(self,
              domain='real',
              epochs=1000,
              ckpt_interval=50,
              max_steps_per_epoch=5000):

        """
        DDPM training
        """

        data = Toyzero(TOYZERO_DIR,
                       partition='train',
                       domain=domain)

        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        self.ddpm.cuda()
        optim = torch.optim.Adam(self.ddpm.parameters(), lr=1e-4)

        ema_loss = 5.0
        for epoch in range(epochs):
            num_steps = min(max_steps_per_epoch, len(dataloader))
            loader_iter = iter(dataloader)
            for step in tqdm(range(num_steps)):
                img, _ = next(loader_iter)
                optim.zero_grad()
                img = img.cuda()
                loss = self.ddpm(img)
                loss.backward()
                optim.step()

                ema_loss *= 0.001
                ema_loss += 0.999 * loss.item()

            print(f"epoch {epoch:05} loss {ema_loss}")
            if (epoch + 1) % ckpt_interval == 0:
                torch.save(self.ddpm.state_dict(), self.ckpt_dir/f"epc_{epoch}.pt")


    def infer(self,
              epoch,
              sample_n=16,
              record_step=50,
              img_dir=None):

        """
        Use saved model at epoch
        """

        states = torch.load(self.ckpt_dir/f"epc_{epoch}.pt")
        self.ddpm.load_state_dict(states).cuda()

        if self.run_name:
            img_dir = IMG_DIR/self.run_name
        else:
            img_dir = IMG_DIR
        if not img_dir.exists():
            img_dir.mkdir(parents=True)
        print(f'Images will be saved to {img_dir}')

        sample_idx_offset = 0
        t_pad = len(str(self.timesteps))
        s_pad = len(str(sample_n))
        while sample_n > 0:

            bsz = min(sample_n, self.batch_size)
            sample_n -= bsz

            print(f'Generating samples {sample_idx_offset} to {sample_idx_offset + bsz - 1}')
            imgs = self.ddpm.denoise_loop(batch_size=bsz, record_step=record_step)

            for time_idx, img_time in enumerate(imgs):
                for sample_idx, img in enumerate(img_time):
                    s_idx = sample_idx_offset + sample_idx
                    t_idx = time_idx * record_step

                    fname_stem = f"sample-%0{s_pad}d_time-%0{t_pad}d" % (s_idx, t_idx)
                    np.savez(img_dir/f'{fname_stem}.npz',
                             data=img.squeeze().cpu().numpy())

            sample_idx_offset += self.batch_size


if __name__ == "__main__":
    fire.Fire(DDPMRun)
