import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import os

class Data(IterableDataset):
  def __init__(self, hf_dataset):
    self.dataset = hf_dataset

  def __iter__(self):
    for item in self.dataset:
        image = item['image'].convert('RGB')
        label = item['label']
        yield image,label


def prep_data(dataset):
    val_batch_size = 1000
    data = Data(dataset)
    DL = DataLoader(data,val_batch_size,shuffle=False,pin_memory=False, num_workers = 0)
    return DL