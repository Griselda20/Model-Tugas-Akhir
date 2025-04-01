import os
import shutil

def reorganize_dataset(train_path):
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    for class_name in classes:
        class_path = os.path.join(train_path, class_name)
        images_path = os.path.join(class_path, 'images')
        
        if os.path.exists(images_path):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for image_file in os.listdir(images_path):
                if os.path.splitext(image_file)[1].lower() in image_extensions:
                    src = os.path.join(images_path, image_file)
                    dst = os.path.join(class_path, image_file)
                    
                    # Hindari overwrite file
                    original_name = image_file
                    new_name = None
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(image_file)
                        counter = 1
                        while os.path.exists(dst):
                            new_name = "{}_{}{}".format(base, counter, ext)
                            dst = os.path.join(class_path, new_name)
                            counter += 1
                    
                    shutil.move(src, dst)
                    
                    # Tampilkan pesan rename/pindah
                    if new_name:
                        print("[Renamed] {} -> {}/{}".format(original_name, class_name, new_name))
                    else:
                        print("[Moved] {} -> {}/{}".format(original_name, class_name, original_name))
            
            # Hapus folder images jika kosong
            if not os.listdir(images_path):
                os.rmdir(images_path)
                print("[Cleaned] Removed empty folder: {}".format(images_path))
        
        # Hapus file class_x_boxes.txt
        txt_file = os.path.join(class_path, "{}_boxes.txt".format(class_name))  # Contoh: class_1_boxes.txt
        if os.path.exists(txt_file):
            os.remove(txt_file)
            print("[Deleted] Removed {}".format(txt_file))

# Jalankan fungsi
reorganize_dataset('/home/tasi2425111/restructure-resized-tiny-imagenet-200/train')