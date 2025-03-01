import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode

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

transform = None

trainset = TrainTinyImageNetDataset(id=id_dict, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

import os
from PIL import Image

# Load class names from words.txt
class_names = {}
with open('/home/tasi2425111/ImageNet/words.txt', 'r') as f:
    for line in f:
        class_id, class_name = line.strip().split('\t')
        class_names[class_id] = class_name


def save_one_image_per_class(dataset, id_dict, class_names, save_dir='images_2', num_classes=10):
    os.makedirs(save_dir, exist_ok=True)
    class_images = {}
    class_labels = {}
    
    # Batasi hanya 10 kelas pertama
    selected_classes = list(id_dict.values())[:num_classes]
    
    for image, label in dataset:
        if label in selected_classes and label not in class_images:
            class_images[label] = image
            class_labels[label] = list(id_dict.keys())[list(id_dict.values()).index(label)]  # Get class ID from label
        if len(class_images) == num_classes:
            break
    
    for class_id, img in class_images.items():
        class_id_str = class_labels[class_id]
        class_name = class_names.get(class_id_str, "Unknown")
        img_pil = transforms.ToPILImage()(img)
        img_pil.save(os.path.join(save_dir, f'{class_name}.png'))

# Call the function to save one image per class (only 10 classes)
save_one_image_per_class(trainset, id_dict, class_names, num_classes=10)