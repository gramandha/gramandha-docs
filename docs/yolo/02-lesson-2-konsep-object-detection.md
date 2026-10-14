# Lesson 2 — Konsep Dasar Object Detection

**Tujuan:** Memahami cara kerja deteksi objek serta mampu mengevaluasi hasil prediksi model secara kuantitatif.

**Tujuan belajar.** Peserta mampu (1) menjelaskan terminologi deteksi objek, (2) mengonversi format bounding box, (3) menghitung IoU, (4) menjelaskan dan mengimplementasikan NMS, (5) menghitung precision, recall, dan menjelaskan AP serta mAP, dan (6) menganalisis jenis kesalahan deteksi.

## 2.1 Terminologi utama

| Istilah | Arti |
| --- | --- |
| Bounding box | Persegi panjang yang melingkupi objek |
| Class label | Nama kategori objek (misalnya `gate` atau `bola`) |
| Class score | Skor model untuk suatu kelas |
| Confidence score | Tingkat keyakinan prediksi; cara perhitungannya bergantung pada model |
| Objectness score | Skor "ada objek atau tidak" pada model yang memakainya (misalnya YOLOv3 hingga YOLOv7). Tidak semua YOLO modern memakainya |
| Confidence threshold | Batas minimum skor agar prediksi dipertahankan |

Perbedaan dengan tugas lain: **klasifikasi** hanya menjawab "apa isi gambar", **deteksi** menjawab "apa dan di mana", **segmentasi** menjawab "piksel mana milik objek apa".

## 2.2 Representasi bounding box

Dua format yang paling umum:

- `xyxy`: `(x1, y1, x2, y2)` — sudut kiri atas dan sudut kanan bawah.
- `xywh`: `(x_tengah, y_tengah, lebar, tinggi)` — pusat dan ukuran.

Koordinat dapat **absolut** (piksel) atau **ternormalisasi** (0–1, dibagi lebar dan tinggi gambar). Label YOLO memakai `xywh` ternormalisasi.

```python
def xywh_to_xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2

def xyxy_to_xywh(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

def norm_to_pixel(xc, yc, w, h, img_w, img_h):
    return xc * img_w, yc * img_h, w * img_w, h * img_h
```

Contoh: gambar 640×480, label ternormalisasi `0.5 0.5 0.25 0.5` berarti kotak berpusat di (320, 240) dengan lebar 160 dan tinggi 240 piksel.

## 2.3 Anchor-based dan anchor-free

- **Anchor-based**: model memprediksi koreksi (offset) terhadap sejumlah kotak acuan (anchor) berukuran tetap pada tiap lokasi. Dipakai pada YOLOv2 hingga YOLOv7.
- **Anchor-free**: model memprediksi posisi atau jarak ke sisi kotak langsung dari titik pada feature map, tanpa daftar anchor. Dipakai misalnya pada YOLOv8 dan penerusnya.

Jangan menyimpulkan bahwa semua YOLO memakai pendekatan yang sama. Periksa dokumentasi tiap model.

## 2.4 Intersection over Union (IoU)

IoU mengukur kemiripan dua kotak: luas irisan dibagi luas gabungan. Nilainya antara 0 (tidak beririsan) dan 1 (identik).

```python
def iou(a, b):
    """a, b dalam format xyxy."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    luas_a = (a[2] - a[0]) * (a[3] - a[1])
    luas_b = (b[2] - b[0]) * (b[3] - b[1])
    union = luas_a + luas_b - inter
    return inter / union if union > 0 else 0.0

print(iou((0, 0, 10, 10), (5, 5, 15, 15)))   # 25 / 175 = 0.1428...
```

Uji kasus: kotak identik (1.0), tidak beririsan (0.0), satu kotak berada di dalam kotak lain.

## 2.5 Non-Maximum Suppression (NMS)

Model sering menghasilkan banyak kotak yang saling tumpang tindih untuk satu objek. NMS memilih kotak terbaik dan membuang duplikatnya.

Algoritma untuk satu kelas:

1. Buang prediksi dengan skor di bawah *confidence threshold*.
2. Urutkan prediksi dari skor tertinggi.
3. Ambil prediksi teratas, simpan sebagai hasil.
4. Buang semua prediksi lain yang IoU-nya terhadap kotak tersebut melebihi *IoU threshold*.
5. Ulangi sampai tidak ada prediksi tersisa.

```python
def nms(kotak, skor, iou_thr=0.5, conf_thr=0.25):
    idx = [i for i, s in enumerate(skor) if s >= conf_thr]
    idx.sort(key=lambda i: skor[i], reverse=True)
    hasil = []
    while idx:
        terbaik = idx.pop(0)
        hasil.append(terbaik)
        idx = [i for i in idx if iou(kotak[terbaik], kotak[i]) < iou_thr]
    return hasil
```

Pengaruh parameter: `conf_thr` rendah menghasilkan lebih banyak deteksi (recall naik, precision turun). `iou_thr` terlalu rendah dapat membuang objek berdekatan; terlalu tinggi menyisakan duplikat. Beberapa model modern (misalnya YOLOv10 dan jalur *end-to-end* pada YOLO26) dapat menghasilkan prediksi tanpa NMS; ini dibahas di Lesson 3.

## 2.6 Metrik evaluasi

Sebuah prediksi dihitung benar (**True Positive, TP**) bila kelasnya benar dan IoU terhadap kotak sebenarnya (ground truth) melebihi ambang tertentu (misalnya 0,5). Prediksi lainnya adalah **False Positive (FP)**. Objek sebenarnya yang tidak terdeteksi adalah **False Negative (FN)**.

- **Precision** = TP / (TP + FP) — dari semua deteksi, berapa yang benar.
- **Recall** = TP / (TP + FN) — dari semua objek nyata, berapa yang terdeteksi.
- **F1-score** = 2 × P × R / (P + R).
- **Precision–Recall curve**: precision terhadap recall saat *confidence threshold* diubah.
- **Average Precision (AP)**: luas di bawah kurva PR untuk satu kelas.
- **mAP@0.5**: rata-rata AP semua kelas dengan ambang IoU 0,5.
- **mAP@0.5:0.95**: rata-rata mAP pada ambang IoU 0,50 hingga 0,95 dengan langkah 0,05; lebih ketat karena menuntut lokalisasi presisi.

Hitung manual contoh kecil:

```python
tp, fp, fn = 8, 2, 4
precision = tp / (tp + fp)         # 0.8
recall = tp / (tp + fn)            # 0.667
f1 = 2 * precision * recall / (precision + recall)
print(precision, recall, round(f1, 3))
```

## 2.7 Analisis kesalahan

| Jenis | Penyebab umum | Tindakan |
| --- | --- | --- |
| False positive | Latar belakang mirip objek, threshold terlalu rendah | Tambah data negatif, naikkan threshold |
| False negative | Objek kecil, terhalang, pencahayaan buruk | Tambah data, resolusi lebih tinggi, augmentasi |
| Salah kelas | Kelas mirip, label tidak konsisten | Perbaiki label, tambah contoh pembeda |
| Kotak kurang pas | Kualitas anotasi, resolusi rendah | Perketat anotasi, periksa mAP@0.5:0.95 |

## 2.8 Praktik utama

Siapkan 5 gambar kecil (atau buat data sintetis) dengan ground truth dan prediksi tulisan tangan. Tugas:

1. Hitung IoU tiap pasangan prediksi dan ground truth.
2. Tandai TP, FP, FN pada ambang IoU 0,5.
3. Hitung precision, recall, dan F1.
4. Jalankan NMS pada prediksi yang tumpang tindih dan bandingkan sebelum dan sesudahnya.

## Cek pemahaman

1. Mengapa mAP@0.5:0.95 biasanya lebih rendah daripada mAP@0.5?
2. Apa efek menurunkan confidence threshold pada precision dan recall?
3. Mengapa objek yang saling berdekatan dapat hilang bila IoU threshold NMS terlalu rendah?
4. Mengapa tidak semua YOLO memiliki objectness score?

## Kesalahan umum

- Mencampur format `xyxy` dan `xywh`.
- Lupa mengalikan koordinat ternormalisasi dengan ukuran gambar.
- Menganggap mAP tinggi berarti model aman di lapangan, padahal data uji tidak representatif.

## Tugas

Kumpulkan hasil Praktik 2.8 ke dalam tabel dan tulis analisis satu paragraf: jenis kesalahan apa yang paling dominan, dan perubahan threshold apa yang akan Anda coba.
