import pytorch_lightning as pl 

from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
from torch_geometric.loader import DataLoader
import os

# this is so that CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 

class WaymoDataModuleCtRLSim(pl.LightningDataModule):

    def __init__(self,
                 train_batch_size,
                 val_batch_size,
                 num_workers,
                 pin_memory,
                 persistent_workers,
                 dataset_cfg):
        super(WaymoDataModuleCtRLSim, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size 
        self.num_workers = num_workers
        self.pin_memory = pin_memory 
        self.persistent_workers = persistent_workers
        self.cfg_dataset = dataset_cfg
        

    def setup(self, stage):
        self.train_dataset = CtRLSimDataset(self.cfg_dataset, split_name='train')
        self.val_dataset = CtRLSimDataset(self.cfg_dataset, split_name='val') 


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True,
                          worker_init_fn=worker_init_fn)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False)