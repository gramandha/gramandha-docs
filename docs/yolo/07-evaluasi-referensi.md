# Evaluasi Akhir dan Referensi

## Hasil akhir pembelajaran

Setelah menyelesaikan keenam lesson, peserta diharapkan mampu:

1. Menjelaskan dasar object detection dan arsitektur YOLO secara konseptual.
2. Menjalankan inferensi menggunakan model pretrained.
3. Melatih dan mengevaluasi YOLO pada dataset kustom.
4. Membandingkan model berdasarkan akurasi, latensi, dan kebutuhan komputasi.
5. Mengekspor serta mengoptimalkan model untuk perangkat target.
6. Mengintegrasikan YOLO ke ROS 2 untuk mendukung aplikasi robotika, misalnya pendeteksian gate dan bola.

## Prinsip evaluasi

Model yang lebih baru tidak otomatis lebih baik untuk setiap kasus penggunaan. Pemilihan model akhir harus didasarkan pada eksperimen yang terkontrol, hasil evaluasi pada data yang representatif, serta kemampuan deployment pada perangkat target.

## Rubrik penilaian

| Aspek | Bobot | Indikator lulus |
| --- | --- | --- |
| Fondasi (Lesson 1–2) | 15% | IoU, NMS, dan metrik dihitung dengan benar; tensor dibentuk dengan dimensi tepat |
| Pemahaman arsitektur (Lesson 3) | 15% | Ringkasan evolusi akurat, merujuk sumber resmi, tanpa generalisasi keliru |
| Inferensi dan pengukuran (Lesson 4) | 15% | Prosedur FPS benar (pemanasan, pengulangan, konfigurasi dicatat) |
| Training dan evaluasi (Lesson 5) | 25% | Dataset tanpa kebocoran, perbandingan adil, analisis kesalahan per kelas dan kondisi |
| Optimasi dan deployment (Lesson 6) | 20% | Ekspor tervalidasi, benchmark lengkap, node ROS 2 berjalan |
| Dokumentasi dan presentasi | 10% | Laporan jelas, eksperimen dapat direproduksi |

## Proyek akhir

**Sistem deteksi gate dan bola pada robot.** Peserta membangun alur lengkap: dataset kustom, pelatihan dan perbandingan YOLO11 dengan YOLO26, ekspor serta optimasi, integrasi ROS 2 Jazzy, dan pengujian sistem. Laporan akhir memuat:

1. Deskripsi dataset (jumlah gambar per kelas, sumber, kondisi, pembagian per sesi).
2. Konfigurasi dan hasil training kedua model pada pembagian data yang sama.
3. Tabel metrik (mAP, precision, recall) dan analisis kesalahan.
4. Hasil benchmark per format dan presisi pada perangkat target.
5. Hasil pengujian sistem ROS 2 (laju, latensi, sumber daya, ketahanan).
6. Keputusan model akhir beserta alasan dan keterbatasannya.

## Pemecahan masalah cepat

| Gejala | Periksa |
| --- | --- |
| Training berjalan dengan dataset kosong | Path di `data.yaml`, struktur `images/` dan `labels/` |
| mAP sangat rendah | Urutan kelas, kualitas label, jumlah data |
| Loss validasi naik | Overfitting; tambah data atau augmentasi |
| Hasil ekspor berbeda jauh | Praproses, pascaproses, format keluaran (NMS atau tanpa NMS) |
| Out of memory | Turunkan `batch` atau `imgsz` |
| FPS jauh di bawah harapan | Pemanasan, perangkat, resolusi, ukuran model, pipeline end-to-end |
| Node ROS 2 tidak menemukan `ultralytics` | Virtual environment dan `PYTHONPATH` |

## Referensi

- [Dokumentasi Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) — fitur, tugas, dan contoh penggunaan YOLO26.
- [Dokumentasi Ultralytics](https://docs.ultralytics.com/) — mode predict, train, val, export, dan panduan dataset.
- [Dokumentasi ROS 2 Jazzy](https://docs.ros.org/en/jazzy/) — dasar node, topic, dan alat command line.
- Dokumentasi PyTorch dan OpenCV — rujukan pemasangan dan API.
- Paper dan dokumentasi resmi tiap versi YOLO (v1 hingga YOLO26) — sumber primer untuk detail arsitektur.

Tautan dan nama format ekspor dapat berubah antar versi. Verifikasi ke dokumentasi resmi sebelum pelatihan berlangsung.
