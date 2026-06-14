import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import os

class Data(IterableDataset):
  def __init__(self, hf_dataset, processor):
    self.dataset = hf_dataset
    self.processor = processor

  def __iter__(self):
    for item in self.dataset:
        image = self.processor(images = item['image'].convert('RGB'), return_tensors = "pt")
        label = item['label']
        image['pixel_values'] = image['pixel_values'].squeeze(1)
        yield image,label


def prep_data(dataset, processor):
    val_batch_size = 1000
    data = Data(dataset,processor)
    DL = DataLoader(data,val_batch_size,shuffle=False,pin_memory=False, num_workers = 0)
    return DL