import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
from PIL import Image

batch_size = 64

id_dict = {}
for i, line in enumerate(open('/home/tasi2425111/ImageNet/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/train/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[5]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/val/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/home/tasi2425111/ImageNet/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class RealTestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/test/*.JPEG")  # Ubah path ke folder test
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        # Karena tidak ada label di folder test, kita bisa mengembalikan -1 atau None
        label = -1  # Atau None jika tidak ada label
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224)
])

# Fungsi untuk resize dan timpa gambar
def resize_and_overwrite_images(dataset):
    for img_path in dataset.filenames:
        image = Image.open(img_path)
        image = transform(image)
        image.save(img_path)

# Resize dan timpa gambar pada dataset train
trainset = TrainTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(trainset)

# Resize dan timpa gambar pada dataset test
testset = TestTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(testset)

# Resize dan timpa gambar pada dataset test
realtestset = RealTestTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(realtestset)

import shutil

# Path folder yang akan di-zip
folder_to_zip = '/home/tasi2425111/ImageNet'

# Nama file zip yang akan dibuat
zip_filename = '/home/tasi2425111/ImageNet_Resize.zip'

# Membuat file zip
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', folder_to_zip)

print(f"Folder {folder_to_zip} telah di-zip dan disimpan sebagai {zip_filename}")