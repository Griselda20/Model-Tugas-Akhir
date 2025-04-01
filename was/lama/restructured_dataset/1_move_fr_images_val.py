import os
import shutil

# Path awal dan target
base_dir = '/home/tasi2425111/restructure-resized-tiny-imagenet-200/val'
src_dir = os.path.join(base_dir, 'images')

# Loop semua file di folder images
for filename in os.listdir(src_dir):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(base_dir, filename)

    # Pindahkan file ke folder val/
    if os.path.isfile(src_path):
        shutil.move(src_path, dst_path)

# Hapus folder images jika sudah kosong
if not os.listdir(src_dir):
    os.rmdir(src_dir)

print("Semua file berhasil dipindah dan folder 'images/' dihapus.")
