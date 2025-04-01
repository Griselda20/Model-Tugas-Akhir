import os
import shutil

# Tentukan direktori dataset
dataset_dir = '/home/tasi2425111/restructure-resized-tiny-imagenet-200/val'  # Ganti dengan path ke folder dataset Anda
annotations_file = '/home/tasi2425111/restructure-resized-tiny-imagenet-200/val/val_annotations.txt'  # Ganti dengan path ke file val_annotations.txt

# Membaca file anotasi yang berisi nama gambar dan kelasnya
with open(annotations_file, 'r') as f:
    annotations = f.readlines()

# Loop melalui setiap baris di file anotasi
for line in annotations:
    # Pisahkan nama file dan label kelas dari setiap baris
    parts = line.split()
    image_filename = parts[0]  # Nama file gambar
    class_id = parts[1]  # ID kelas
    
    # Tentukan path lengkap gambar
    image_path = os.path.join(dataset_dir, image_filename)
    
    # Tentukan direktori tujuan berdasarkan kelas
    class_dir = os.path.join(dataset_dir, class_id)
    
    # Jika folder untuk kelas belum ada, buat folder tersebut
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    # Tentukan path tujuan untuk memindahkan gambar
    destination_path = os.path.join(class_dir, image_filename)
    
    # Pindahkan gambar ke folder kelas yang sesuai
    shutil.move(image_path, destination_path)
    
    # Perbaikan: menggunakan format() untuk mencetak pesan
    print("Moved {} to {} folder".format(image_filename, class_id))

print("Folder structure creation and image sorting completed!")
