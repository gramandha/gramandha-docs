# Lesson 3 — Arsitektur dan Evolusi YOLO (YOLOv1–YOLO26)

**Tujuan:** Memahami perkembangan arsitektur YOLO, inovasi utama setiap generasi, dan pengaruhnya terhadap akurasi, kecepatan inferensi, serta deployment.

**Tujuan belajar.** Peserta mampu (1) menjelaskan filosofi one-stage detection, (2) menyebutkan peran backbone, neck, dan head, (3) merangkum inovasi utama tiap generasi YOLO, (4) menjelaskan perbedaan jalur NMS dan end-to-end, dan (5) membandingkan model secara adil.

> **Catatan verifikasi.** Detail arsitektur, dukungan tugas, dan fitur inferensi harus diverifikasi pada dokumentasi resmi atau paper masing-masing model. Tidak semua YOLO memakai grid, anchor boxes, attention, atau mekanisme NMS yang sama. Ringkasan di bawah adalah peta belajar, bukan pengganti sumber primer.

## 3.1 Konsep fundamental

- **One-stage detection**: lokasi dan kelas objek diprediksi dalam satu alur inferensi, tanpa tahap pengusulan region terpisah seperti pada detektor dua tahap.
- **Backbone**: jaringan ekstraksi fitur dari citra.
- **Neck**: menggabungkan fitur dari berbagai skala (misalnya struktur ala feature pyramid) agar objek kecil dan besar terdeteksi.
- **Detection head**: menghasilkan prediksi kotak dan kelas dari fitur hasil neck.
- **Bounding box regression dan klasifikasi**: dua komponen utama yang dipelajari model, masing-masing dengan fungsi loss sendiri.
- **Penentuan target training (label assignment)**: aturan menentukan prediksi mana yang dipasangkan dengan ground truth mana saat menghitung loss.
- **Grid S × S**: bagian dari formulasi awal YOLOv1, **bukan** mekanisme universal semua versi modern.
- **NMS dan deteksi end-to-end**: sebagian model memerlukan NMS sebagai post-processing; sebagian lain dirancang menghasilkan prediksi tanpa NMS.

Diagram alir sederhana:

```text
Gambar → Backbone → Neck → Head → prediksi mentah → (NMS bila dipakai) → deteksi akhir
```

## 3.2 Perkembangan arsitektur

| Generasi | Inovasi yang perlu dipelajari |
| --- | --- |
| YOLOv1–YOLOv3 | Formulasi awal berbasis grid, prediksi kotak dan kelas sekaligus, penggunaan Darknet, anchor boxes (YOLOv2–YOLOv3), prediksi multi-skala |
| YOLOv4–YOLOv7 | Pengembangan backbone dan neck, desain berbasis CSPNet, augmentasi Mosaic, E-ELAN pada YOLOv7. Re-parameterization adalah teknik tersendiri dan bukan fitur universal seluruh versi dalam rentang ini |
| YOLOv8–YOLOv10 | Anchor-free detection pada YOLOv8; Programmable Gradient Information (PGI) dan Generalized Efficient Layer Aggregation Network (GELAN) pada YOLOv9; consistent dual assignments dan desain end-to-end tanpa NMS pada YOLOv10 |
| YOLO11 | Blok C3k2 dan C2PSA, termasuk penggunaan mekanisme attention pada bagian arsitektur tertentu |
| YOLO12 | Pendekatan attention-centric: Area Attention dan R-ELAN; perhatikan kebutuhan komputasi dan karakteristik training |
| YOLO26 | End-to-end, head lebih ringan, regresi bounding box tanpa Distribution Focal Loss (DFL), jalur inferensi tanpa NMS yang dapat dipilih |

### YOLO26 secara lebih rinci

Berdasarkan dokumentasi resmi Ultralytics YOLO26 (halaman `docs.ultralytics.com/models/yolo26`):

- **Arsitektur dua kepala untuk deteksi.** Kepala *one-to-many* (default) menghasilkan keluaran yang memerlukan NMS dan umumnya sedikit lebih akurat. Kepala *one-to-one* (diaktifkan dengan `nms=False`) menghasilkan prediksi langsung tanpa NMS, dengan maksimum 300 deteksi per gambar.
- **Tanpa DFL.** Penghapusan Distribution Focal Loss menyederhanakan kepala deteksi dan ekspor.
- **Resep training.** Memakai optimizer MuSGD (gabungan SGD dan Muon), Progressive Loss, dan STAL (Small-Target-Aware Label Assignment) untuk objek kecil.
- **Keluarga model.** Lima ukuran (n, s, m, l, x) dan beberapa tugas: deteksi, segmentasi instans, segmentasi semantik, estimasi kedalaman, klasifikasi, pose, dan oriented bounding box.
- **Varian arsitektur.** `yolo26-p2.yaml` dan `yolo26-p6.yaml` tersedia sebagai konfigurasi YAML tanpa bobot pretrained resmi.
- **Lisensi.** AGPL-3.0 atau lisensi Enterprise; pertimbangkan implikasinya bila model dipakai di produk.

Angka akurasi dan kecepatan pada dokumentasi diukur pada dataset dan perangkat tertentu (COCO, CPU ONNX, T4 TensorRT). Jangan memakainya sebagai jaminan untuk perangkat dan data Anda.

## 3.3 Analisis komparatif

Dimensi pembandingan:

| Dimensi | Cara mengukur |
| --- | --- |
| Jumlah parameter dan FLOPs | `model.info()` pada Ultralytics |
| Precision, Recall, mAP | `model.val()` pada dataset yang sama |
| Latensi inferensi dan FPS | Ukur sendiri dengan pemanasan dan banyak pengulangan (lihat Lesson 4 dan 6) |
| Penggunaan memori | Pantau RAM/VRAM saat inferensi |
| Kompatibilitas deployment | Uji ekspor ke format target (Lesson 6) |

Aturan perbandingan adil: dataset sama, pembagian data sama, resolusi input sama, perangkat dan pengaturan sama (termasuk apakah NMS dipakai).

## 3.4 Praktik utama: YOLO11 vs YOLO26

Pasang Ultralytics terlebih dahulu (lihat Lesson 4), lalu:

```python
from ultralytics import YOLO

for nama in ["yolo11n.pt", "yolo26n.pt"]:
    model = YOLO(nama)
    print("=== ", nama)
    model.info()          # layer, parameter, GFLOPs
```

Jalankan validasi pada dataset kecil yang sama (misalnya `coco8.yaml` untuk uji cepat):

```python
for nama in ["yolo11n.pt", "yolo26n.pt"]:
    model = YOLO(nama)
    m = model.val(data="coco8.yaml", imgsz=640)
    print(nama, "mAP50-95 =", m.box.map, "mAP50 =", m.box.map50)
```

Catatan: `coco8` hanya berisi 8 gambar sehingga cocok untuk uji alur kerja, bukan untuk kesimpulan akurasi. Gunakan dataset bermakna untuk perbandingan sungguhan.

Isi tabel hasil Anda:

| Model | Parameter | GFLOPs | mAP50 | mAP50-95 | Latensi (ms) | Perangkat |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | | | | | | |
| YOLO26n | | | | | | |

YOLOv8–YOLOv10 dan YOLO12 dapat dipelajari sebagai pembanding sesuai kebutuhan penelitian.

## Cek pemahaman

1. Apa peran backbone, neck, dan head?
2. Mengapa grid S × S tidak boleh dianggap berlaku untuk semua YOLO modern?
3. Apa beda kepala one-to-many dan one-to-one pada YOLO26?
4. Mengapa perbandingan model harus memakai pembagian data dan resolusi yang sama?

## Kesalahan umum

- Menyimpulkan model terbaru pasti terbaik tanpa eksperimen.
- Membandingkan angka dari paper berbeda yang memakai perangkat dan pengaturan berbeda.
- Mencampur hasil dengan NMS dan tanpa NMS tanpa mencatatnya.

## Tugas

Buat satu halaman ringkasan evolusi YOLO dengan format: generasi, inovasi utama, dampak ke akurasi/kecepatan/deployment, dan sumber resmi yang Anda rujuk (tautan paper atau dokumentasi).
