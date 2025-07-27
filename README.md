# 🗑️ Trash Classification System - Computer Vision Realtime

Sistem klasifikasi sampah menggunakan deep learning dan computer vision untuk mendeteksi jenis sampah secara realtime melalui webcam.

## 🚀 Fitur Utama

- **Klasifikasi Realtime**: Deteksi sampah menggunakan webcam secara langsung
- **6 Kategori Sampah**: Plastik, Logam/Besi, Kertas, Kaca, Organik, Kardus
- **Deep Learning**: Menggunakan CNN dengan transfer learning (MobileNetV2)
- **Interface Interaktif**: Menu yang mudah digunakan
- **Rekomendasi Penanganan**: Saran cara membuang/mendaur ulang sampah

## 📋 Requirements

### Dependencies
Pastikan Anda sudah menginstall dependencies berikut:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python Pillow
```

Atau install menggunakan requirements.txt:
```bash
pip install -r requirements.txt
```

## 🛠️ Cara Instalasi dan Penggunaan

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan program**:
   ```bash
   python main.py
   ```
3. **Pilih menu 2** untuk melatih model CNN (wajib untuk fitur realtime)
4. **Pilih menu 4** untuk klasifikasi realtime dengan webcam

## 📖 Menu Program

```
🗑️ PROGRAM KLASIFIKASI SAMPAH
==================================================

Pilih menu:
1. Latih model dengan data sampel (fitur)
2. Latih model CNN dengan data gambar (WAJIB untuk realtime)
3. Klasifikasi sampah interaktif (input manual)
4. Klasifikasi sampah realtime (webcam)
5. Keluar
```

## 🎥 Fitur Realtime Webcam

### Kontrol:
- **'q'**: Keluar dari mode realtime
- **'s'**: Screenshot dan analisis

### Informasi yang Ditampilkan:
- Kategori sampah secara realtime
- Confidence score (0-100%)
- FPS counter
- Rekomendasi penanganan

## 🎯 Kategori Sampah

1. **🔵 Plastik**: Botol plastik, kantong plastik
2. **⚫ Logam/Besi**: Kaleng, besi, aluminium  
3. **📄 Kertas**: Kertas, koran, majalah
4. **💎 Kaca**: Botol kaca, gelas
5. **🌱 Organik**: Sisa makanan, daun
6. **📦 Kardus**: Kotak kardus, karton

---

**Created with ❤️ for a cleaner environment 🌍**