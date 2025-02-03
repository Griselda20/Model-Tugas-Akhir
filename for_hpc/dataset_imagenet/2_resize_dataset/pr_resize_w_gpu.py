import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import sys  # Import modul sys untuk flush output

# Periksa ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")
sys.stdout.flush()  # Memaksa flush output

batch_size = 64

# Membaca file wnids.txt untuk membuat id_dict
print("Membaca file wnids.txt...")
sys.stdout.flush()  # Memaksa flush output
id_dict = {}
for i, line in enumerate(open('/home/tasi2425111/ImageNet/wnids.txt', 'r')):
    id_dict[line.replace('\n', '')] = i
print("wnids.txt berhasil dibaca.")
sys.stdout.flush()  # Memaksa flush output

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        print("Mempersiapkan dataset train...")
        sys.stdout.flush()  # Memaksa flush output
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/train/*/*.JPEG")
        self.transform = transform
        self.id_dict = id
        print(f"Dataset train berhasil dimuat dengan {len(self.filenames)} gambar.")
        sys.stdout.flush()  # Memaksa flush output

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[5]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor)).to(device)
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        print("Mempersiapkan dataset test...")
        sys.stdout.flush()  # Memaksa flush output
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/val/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/home/tasi2425111/ImageNet/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
        print(f"Dataset test berhasil dimuat dengan {len(self.filenames)} gambar.")
        sys.stdout.flush()  # Memaksa flush output

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor)).to(device)
        return image, label

class RealTestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        print("Mempersiapkan dataset real test...")
        sys.stdout.flush()  # Memaksa flush output
        self.filenames = glob.glob("/home/tasi2425111/ImageNet/test/*.JPEG")
        self.transform = transform
        self.id_dict = id
        print(f"Dataset real test berhasil dimuat dengan {len(self.filenames)} gambar.")
        sys.stdout.flush()  # Memaksa flush output

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        label = -1  # Atau None jika tidak ada label
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor)).to(device)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224)
])

# Fungsi untuk resize dan timpa gambar
def resize_and_overwrite_images(dataset, dataset_name, start_index=0):  
    print(f"Memulai proses resize dan overwrite untuk dataset {dataset_name}...")  
    sys.stdout.flush()  # Memaksa flush output  
    total_images = len(dataset.filenames)  
    
    # Mulai iterasi dari start_index  
    for i in range(start_index, total_images):  
        img_path = dataset.filenames[i]  
        try:  
            # Coba buka gambar  
            image = Image.open(img_path)  
            
            # Konversi ke mode RGB jika gambar memiliki mode RGBA  
            if image.mode == 'RGBA':  
                image = image.convert('RGB')  
            
            # Resize gambar  
            image = transform(image)  
            
            # Simpan gambar  
            image.save(img_path)  
            
            if (i + 1) % 100 == 0:  # Cetak log setiap 100 gambar  
                print(f"Proses {i + 1}/{total_images} gambar selesai.")  
                sys.stdout.flush()  # Memaksa flush output  
        except UnidentifiedImageError:  
            # Tangani file yang tidak valid  
            print(f"Peringatan: File {img_path} tidak dapat diidentifikasi sebagai gambar. Melewati file ini.")  
            sys.stdout.flush()  # Memaksa flush output  
        except Exception as e:  
            # Tangani error lainnya  
            print(f"Error saat memproses {img_path}: {e}")  
            sys.stdout.flush()  # Memaksa flush output  
    print(f"Proses resize dan overwrite untuk dataset {dataset_name} selesai.")  
    sys.stdout.flush()  # Memaksa flush output

# Resize dan timpa gambar pada dataset train
print("Memulai proses resize untuk dataset train...")
sys.stdout.flush()  # Memaksa flush output
trainset = TrainTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(trainset, "train", start_index=294300)

# Resize dan timpa gambar pada dataset test
print("Memulai proses resize untuk dataset test...")
sys.stdout.flush()  # Memaksa flush output
testset = TestTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(testset, "test")

# Resize dan timpa gambar pada dataset real test
print("Memulai proses resize untuk dataset real test...")
sys.stdout.flush()  # Memaksa flush output
realtestset = RealTestTinyImageNetDataset(id=id_dict)
resize_and_overwrite_images(realtestset, "real test")

import shutil

# Path folder yang akan di-zip
folder_to_zip = '/home/tasi2425111/ImageNet'

# Nama file zip yang akan dibuat
zip_filename = '/home/tasi2425111/ImageNet_Resize.zip'

# Membuat file zip
print(f"Membuat file zip dari folder {folder_to_zip}...")
sys.stdout.flush()  # Memaksa flush output
shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', folder_to_zip)
print(f"Folder {folder_to_zip} telah di-zip dan disimpan sebagai {zip_filename}")
sys.stdout.flush()  # Memaksa flush output

print("Semua proses selesai.")
sys.stdout.flush()  # Memaksa flush output