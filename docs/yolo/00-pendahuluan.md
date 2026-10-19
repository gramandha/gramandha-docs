# Modul Pembelajaran YOLO: Dari Dasar hingga Deployment Edge AI

## Pendahuluan

Modul ini membawa peserta dari dasar Python dan pengolahan citra sampai mampu melatih model YOLO pada dataset sendiri, mengoptimalkannya, dan menjalankannya di perangkat edge serta ROS 2. Modul terdiri dari 6 lesson yang dirancang berurutan, masing-masing sekitar 3 jam (teori dan praktik).

**Tujuan akhir.** Setelah menyelesaikan modul, peserta mampu:

1. Menjelaskan dasar object detection dan arsitektur YOLO secara konseptual.
2. Menjalankan inferensi menggunakan model pretrained.
3. Melatih dan mengevaluasi YOLO pada dataset kustom.
4. Membandingkan model berdasarkan akurasi, latensi, dan kebutuhan komputasi.
5. Mengekspor serta mengoptimalkan model untuk perangkat target.
6. Mengintegrasikan YOLO ke ROS 2 untuk aplikasi robotika, dengan studi kasus robot yang mendeteksi gate dan bola.

## Prasyarat

- Laptop dengan Ubuntu 24.04 (disarankan) atau sistem operasi lain yang mendukung Python 3.10 atau lebih baru.
- RAM minimal 8 GB dan ruang disk 20 GB. GPU NVIDIA sangat membantu untuk pelatihan tetapi tidak wajib; Google Colab dapat dipakai sebagai alternatif.
- Kemampuan dasar menjalankan perintah di terminal.
- Untuk Lesson 6: pemahaman dasar ROS 2 (node, topic, package). Rujuk modul pelatihan ROS 2 Jazzy.

## Peta lesson

| Lesson | Topik | Keluaran yang diverifikasi |
| --- | --- | --- |
| 1 | Prasyarat dasar | Gambar terbaca OpenCV dan berubah menjadi tensor PyTorch |
| 2 | Konsep object detection | Fungsi IoU dan evaluasi precision/recall sederhana |
| 3 | Arsitektur dan evolusi YOLO | Tabel perbandingan YOLO11 dan YOLO26 pada kondisi sama |
| 4 | Inferensi model pretrained | Deteksi gambar, video, webcam beserta FPS terukur |
| 5 | Pelatihan dataset kustom | Model terlatih dan laporan evaluasi per kelas |
| 6 | Optimasi dan deployment | Model ekspor tervalidasi dan node ROS 2 berjalan |

## Cara memakai modul

Setiap lesson berisi: tujuan belajar, teori, praktik bertahap, cek pemahaman, kesalahan umum, dan tugas. Kerjakan secara berurutan karena setiap lesson memakai hasil lesson sebelumnya.

## Prinsip kerja di sepanjang modul

- **Bandingkan secara adil.** Perbandingan model hanya valid bila dataset, pembagian data, resolusi input, dan perangkat sama.
- **Ukur, jangan berasumsi.** Klaim FPS, akurasi, dan dukungan fitur harus diverifikasi pada perangkat dan versi perangkat lunak Anda.
- **Model terbaru belum tentu terbaik.** Pemilihan model akhir ditentukan oleh eksperimen terkontrol, data yang representatif, dan kemampuan deployment.
- **Verifikasi ke sumber resmi.** Detail arsitektur dan API dapat berubah antar versi, sehingga rujuk dokumentasi resmi dan paper masing-masing model.

## Menyiapkan environment (dipakai di semua lesson)

```bash
$ python3 -m venv ~/yolo_env
$ source ~/yolo_env/bin/activate
$ pip install --upgrade pip
$ pip install numpy opencv-python matplotlib pandas
```

Paket PyTorch dan Ultralytics dipasang di Lesson 4 agar versinya sesuai perangkat Anda.
