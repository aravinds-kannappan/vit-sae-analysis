import torch

from torch.utils.data import IterableDataset, DataLoader


from PIL import Image

import os



class Data(IterableDataset):

  def __init__(self, hf_dataset, processor):

    self.dataset = hf_dataset

    self.processor = processor



  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:
       
      for item in self.dataset:

          image = self.processor(images = item['image'].convert('RGB'), return_tensors = "pt")

          image['pixel_values'] = image['pixel_values'].squeeze(0)

          label = item['label']

          yield image,label
    else:
       
      worker_id = worker_info.id
      num_workers = worker_info.num_workers

      # Splice the dataset such that no data duplication happens among workers
      for idx, item in enumerate(self.dataset):
        if idx % num_workers == worker_id:
          image = self.processor(images = item['image'].convert('RGB'), return_tensors = "pt")

          image['pixel_values'] = image['pixel_values'].squeeze(0)

          label = item['label']

          yield image, label





def prep_data(dataset, processor):

    val_batch_size = 1000

    data = Data(dataset,processor)

    DL = DataLoader(

        data,

        val_batch_size,

        shuffle=False,

        pin_memory=False,

        num_workers = min(16,os.cpu_count())

    )

    return DL