from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire

import torch

from ddpm_tutorial.unet import Unet
from ddpm_tutorial.data import FashionMNIST, CIFAR10, transform, reverse_transform
from ddpm_tutorial.dataset_toyzero import Toyzero
from ddpm_tutorial.ddpm import DDPM

DATA_DIR = Path("./data/")
CKPT_DIR = Path("./checkpoints/")
IMG_DIR = Path("./images/")
TOYZERO_DIR = Path('/data/datasets/LS4GAN/toy-adc_256x256_precropped')

def get_config(dataset):
    """
    Get dataset config
    """

    if dataset == "toyzero":
        data = Toyzero(TOYZERO_DIR, partition='train', domain='real', max_dataset_size=2000)
        return {'data':data, 'channels': 1, 'img_sz': (1, 256, 256)}

    if dataset == "fashion":
        data = FashionMNIST(DATA_DIR, train=True, transform=transform, download=True)
        return {'data': data, 'channels': 1, "img_sz": (1, 32, 32)}

    if dataset == "cifar10":
        data = CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
        return  {"data": data, "channels": 3, "img_sz": (3, 32, 32)}

    raise ValueError(f"{dataset} not supported")


class DDPMRun():
    """
    DDPM model training and inference
    """
    def __init__(self, dataset="cifar10", timesteps=1000, run_name=None):

        assert dataset in ("fashion", "cifar10", "toyzero")
        self.dataset = dataset
        self.timesteps = timesteps

        if run_name:
            self.ckpt_dir = CKPT_DIR/run_name
        else:
            self.ckpt_dir = CKPT_DIR
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        print(f'Checkpoints will be save to {self.ckpt_dir}')


    def train(self, batch_size=256, epochs=1000, ckpt_interval=10):
        """
        DDPM training
        """
        cfg = get_config(self.dataset)
        dataloader = torch.utils.data.DataLoader(
            cfg["data"],
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=self.timesteps, img_sz=cfg["img_sz"])
        ddpm.cuda()
        optim = torch.optim.Adam(ddpm.parameters(), lr=0.0001)

        ema_loss = 5.0

        for epoch in tqdm(range(epochs)):
            for img, _ in dataloader:
                optim.zero_grad()
                img = img.cuda()
                loss = ddpm(img)
                loss.backward()
                optim.step()

                ema_loss *= 0.001
                ema_loss += 0.999 * loss.item()

            print(f"epoch {epoch:05} loss {ema_loss}")
            if (epoch + 1) % ckpt_interval == 0:
                torch.save(ddpm.state_dict(), self.ckpt_dir/f"{self.dataset}_epc_{epoch}.pt")

        torch.save(ddpm.state_dict(), self.ckpt_dir/f"{self.dataset}_epc_{epochs-1}.pt")

    def infer(self, epoch, sample_n=16, batch_size=16, record_step=50, img_dir=None):
        """
            use saved model at epoch
        """
        cfg = get_config(self.dataset)
        model = Unet(dim=32, dim_mults=(1, 2, 4,), channels=cfg["channels"])
        ddpm = DDPM(model, timesteps=self.timesteps, img_sz=cfg["img_sz"])
        states = torch.load(self.ckpt_dir/f"{self.dataset}_epc_{epoch}.pt")
        ddpm.load_state_dict(states)
        ddpm.cuda()

        if not img_dir:
            img_dir = IMG_DIR
        if not img_dir.exists():
            img_dir.mkdir(parents=True)

        sample_idx_offset = 0
        t_pad = len(str(self.timesteps))
        s_pad = len(str(sample_n))
        while sample_n > 0:

            bsz = min(sample_n, batch_size)
            sample_n -= bsz

            imgs = ddpm.denoise_loop(batch_size=bsz, record_step=record_step)

            for time_idx, img_time in enumerate(imgs):
                for sample_idx, img in enumerate(img_time):
                    s_idx = sample_idx_offset + sample_idx
                    t_idx = time_idx * record_step

                    fname_stem = f"{self.dataset}_{self.timesteps}_sample-%0{s_pad}d_time-%0{t_pad}d" % (s_idx, t_idx)
                    if self.dataset == 'toyzero':
                        img = img.squeeze().cpu().numpy()
                        np.savez(img_dir/f'{fname_stem}.npz', data=img)
                    else:
                        img = reverse_transform(img)
                        with open(img_dir/f'{fname_stem}.png', 'wb') as file_pointer:
                            img.save(file_pointer)

            sample_idx_offset += batch_size


if __name__ == "__main__":
    fire.Fire(DDPMRun)
