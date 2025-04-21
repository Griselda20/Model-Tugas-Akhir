# Analisis Performa Model Hybrid CNN-ViT  
**Studi Perbandingan Mobile-Former, CoAtNet, dan Model Modifikasi untuk Klasifikasi Gambar**

## 📌 Deskripsi
Repositori ini merupakan bagian dari tugas akhir berjudul **"Analisis Performa Model Hybrid CNN-ViT dengan Model Mobile-Former, CoAtNet, dan Modifikasi untuk Klasifikasi Gambar dengan Berbagai Ukuran Dataset."**

Penelitian ini bertujuan untuk membandingkan performa tiga arsitektur deep learning, yaitu:  **Mobile-Former**, **CoAtNet**, dan **Model Modifikasi (Mobile-Former - CoAtNet)** dalam klasifikasi gambar menggunakan dataset kecil(Tiny-ImageNet) dan besar(ImageNet-1K), serta mengevaluasi performanya dari segi akurasi (Top-1 Accuracy) dan efisiensi (FLOPs).

## 📂 Struktur File Utama

| File                                  | Deskripsi                                                                |
|---------------------------------------|--------------------------------------------------------------------------|
| `Model_CoAtNet.py`                    | Arsitektur model CoAtNet                                                 |
| `Model_MobileFormer.py`               | Arsitektur model Mobile-Former                                           |
| `Model_Modification.py`               | Model modifikasi berbasis Mobile-Former dan CoAtNet                      |
| `Flops_Evaluation_[model].py`         | Evaluasi efisiensi model dengan menghitung FLOPs                         |
| `Train_[model]_on_Imagenet.py`        | Pelatihan model pada dataset ImageNet                                    |
| `Train_[model]_on_TinyImagenet.py`    | Pelatihan model pada dataset TinyImageNet                                |
| `resize_and_save_to_224.py`           | Preprocessing gambar untuk penyesuaian ukuran input (224x224)            |

## 📊 Hasil
- **Model Modifikasi** berhasil diterapkan pada dataset Tiny-ImageNet dan ImageNet-1K.
- **Model Modifikasi** menunjukkan akurasi yang lebih tinggi dibandingkan Mobile-Former, dengan peningkatan FLOPs yang relatif kecil.
- Evaluasi performa dilakukan berdasarkan **Top-1 Accuracy** dan **efisiensi model (FLOPs)**.


## 👥 Kelompok
**TASI-2425-111**

- 12S21041 – Samuel Christy Angie Sihotang  
- 12S21052 – Griselda  
- 12S21057 – Agnes Theresia Siburian  

Institut Teknologi Del  
Program Studi Sistem Informasi  
Tugas Akhir Tahun Akademik 2024/2025

