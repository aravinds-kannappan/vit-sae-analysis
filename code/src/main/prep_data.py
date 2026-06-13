import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
import os

class Data(Dataset):
  def __init__(self, hf_dataset, transform):
    self.dataset = hf_dataset
    self.transform = transform
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self, index):
     item = self.dataset[index]
     image = self.transform(item['image'].convert('RGB'))
     label = item['label']
     return image,label


def prep_data(dataset, config, corruption_type = None):
    if corruption_type not in ["JPEG", "Gaussian Blur", None]:
        raise ValueError("corruption_type must be 'JPEG', 'Gaussian Blur', or None")
    # Setting up transforms
    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    val_batch_size = 1000
    if not corruption_type:
        transform_val = T.Compose([
            T.Resize(config.dimensions),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset, transform_val)
    elif corruption_type == "Gaussian Blur":
        GB_transform = T.Compose([
            T.Resize(config.dimensions),
            T.GaussianBlur(kernel_size=5, sigma=2.5),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset, GB_transform)
    elif corruption_type == "JPEG":
        jpeg_transforms = T.Compose([
            T.Resize(config.dimensions),
            T.JPEG(quality=5),
            T.ToTensor(),
            T.ToDtype(torch.float32, scale=True),
            normalize
        ])
        data = Data(dataset, jpeg_transforms)
    DL = DataLoader(data,val_batch_size,shuffle=False,pin_memory=False, num_workers = min(8,os.cpu_count()))
    return DL