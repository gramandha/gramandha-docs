# Lesson 5 — Pelatihan Dataset Kustom (Custom Training)

**Tujuan:** Mampu membangun model YOLO untuk mendeteksi objek spesifik berdasarkan dataset yang dikumpulkan sendiri.

**Tujuan belajar.** Peserta mampu (1) mengumpulkan dan mengkurasi data, (2) menganotasi dan memvalidasi label, (3) menyusun dataset dalam format YOLO beserta `data.yaml`, (4) membagi data tanpa kebocoran, (5) melatih dan menyetel model, dan (6) mengevaluasi serta menganalisis kesalahan dan generalisasi.

Contoh kasus di sepanjang lesson: deteksi **gate** dan **bola** oleh robot pada arena. Ganti dengan kasus Anda sendiri bila berbeda.

## 5.1 Pengumpulan dan kurasi data

Prinsip utama: data latih harus mewakili kondisi operasional nyata.

- Kumpulkan gambar dari kondisi yang akan dihadapi: variasi pencahayaan, jarak objek, sudut kamera, posisi robot (diam dan bergerak), dan latar belakang arena.
- Periksa kualitas: buang gambar rusak, terlalu gelap atau buram sehingga objek tidak dapat dikenali manusia, dan duplikat.
- Sertakan **gambar tanpa objek target** (data negatif) agar model belajar tidak mendeteksi sembarangan.
- Catat metadata sesi (tanggal, lokasi, kondisi) untuk pembagian data yang benar di 5.4.

Sebagai pegangan awal, mulai dari ratusan gambar per kelas dan tambahkan sesuai hasil evaluasi. Angka kecukupan data bergantung pada kesulitan kasus, jadi putuskan berdasarkan eksperimen.

Menemukan duplikat persis dengan hash:

```python
import hashlib, pathlib, collections

hasil = collections.defaultdict(list)
for p in pathlib.Path("dataset_mentah").rglob("*.jpg"):
    hasil[hashlib.md5(p.read_bytes()).hexdigest()].append(p)

for h, daftar in hasil.items():
    if len(daftar) > 1:
        print("duplikat:", [str(d) for d in daftar])
```

Hash hanya menangkap duplikat identik. Frame video yang hampir sama harus ditangani lewat pembagian per sesi (5.4).

## 5.2 Anotasi data

Alat yang umum: Roboflow, CVAT, atau LabelImg. Pilih sesuai jenis tugas dan dukungan format ekspor (pastikan dapat mengekspor ke format YOLO).

Pedoman anotasi:

1. Tentukan daftar kelas dan definisi tertulis sebelum mulai (apakah "gate" berarti seluruh gerbang atau hanya tiang dan palangnya? bola yang sebagian tertutup tetap dianotasi?).
2. Kotak harus rapat mengikuti objek, termasuk objek yang sebagian tertutup bila memang akan dideteksi.
3. Konsisten antar penganotasi; lakukan peninjauan silang pada sampel.
4. Validasi label: tidak ada kotak di luar gambar, kelas tidak salah, tidak ada file label yatim.

## 5.3 Format dataset YOLO

Struktur direktori:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/   gambar_001.jpg ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/   gambar_001.txt ...
    ├── val/
    └── test/
```

Setiap file label `.txt` berisi satu baris per objek:

```text
<id_kelas> <x_tengah> <y_tengah> <lebar> <tinggi>
```

Semua koordinat ternormalisasi 0–1. Contoh: `0 0.512 0.430 0.210 0.180`. Gambar tanpa objek memiliki file label kosong atau tanpa file label (ikuti konvensi alat Anda).

`data.yaml`:

```yaml
path: /home/user/dataset     # folder akar dataset
train: images/train
val: images/val
test: images/test

names:
  0: gate
  1: bola
```

Skrip pemeriksa label:

```python
import pathlib

akar = pathlib.Path("dataset")
jumlah_kelas = 2
for txt in (akar / "labels").rglob("*.txt"):
    for no, baris in enumerate(txt.read_text().splitlines(), 1):
        bagian = baris.split()
        if len(bagian) != 5:
            print(txt, no, "format salah"); continue
        k, *nilai = bagian
        nilai = list(map(float, nilai))
        if not (0 <= int(k) < jumlah_kelas):
            print(txt, no, "kelas tidak valid")
        if any(v < 0 or v > 1 for v in nilai):
            print(txt, no, "koordinat di luar 0-1")
```

## 5.4 Pembagian dataset tanpa kebocoran

**Kebocoran data** terjadi bila gambar yang hampir identik muncul di data latih dan data uji sehingga evaluasi tampak lebih bagus daripada kenyataan.

- Bagi berdasarkan **sesi atau video**, bukan acak per frame. Semua frame dari satu sesi masuk ke satu subset saja.
- Pertahankan sebaran kelas dan kondisi yang mirip di tiap subset.
- Simpan data uji (`test`) yang tidak disentuh selama pengembangan. Gunakan `val` untuk penyetelan dan `test` hanya sekali di akhir.

```python
import random, shutil, pathlib

sesi = ["sesi_01", "sesi_02", "sesi_03", "sesi_04", "sesi_05", "sesi_06", "sesi_07", "sesi_08"]
random.seed(42)
random.shuffle(sesi)
split = {"train": sesi[:5], "val": sesi[5:7], "test": sesi[7:]}
print(split)
# lanjutkan dengan menyalin gambar dan label tiap sesi ke folder subset yang sesuai
```

## 5.5 Augmentasi data

Augmentasi memperbanyak variasi secara buatan. Prinsip: harus realistis dan tidak merusak validitas label.

| Augmentasi | Pertimbangan |
| --- | --- |
| Rotasi kecil | Aman bila kamera bisa sedikit miring; rotasi besar dapat membuat kotak tidak pas |
| Skala dan crop | Membantu variasi jarak objek |
| Variasi warna (HSV) | Berguna untuk perubahan pencahayaan dan warna |
| Flip horizontal | Aman bila objek tidak bergantung pada arah kiri-kanan |
| Mosaic | Menggabungkan beberapa gambar; Ultralytics mengaktifkannya secara default |

Ultralytics menerapkan augmentasi saat training lewat argumen seperti `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `scale`, `fliplr`, dan `mosaic`. Nilai default dan daftar argumen dapat berubah antar versi; rujuk halaman konfigurasi training pada dokumentasi resmi.

## 5.6 Training dan fine-tuning

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")        # mulai dari bobot pretrained (transfer learning)
hasil = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,                  # early stopping bila tidak membaik 20 epoch
    seed=0,
    project="runs_ukm",
    name="yolo26n_gate_bola",
)
```

Padanan CLI:

```bash
$ yolo detect train model=yolo26n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16
```

Penjelasan hiperparameter:

| Parameter | Arti | Panduan awal |
| --- | --- | --- |
| Pretrained weights | Titik awal bobot | Selalu mulai dari pretrained pada dataset kecil |
| `epochs` | Berapa kali seluruh data dilewati | Atur cukup besar, dibatasi `patience` |
| `batch` | Jumlah gambar per langkah | Turunkan bila memori GPU habis |
| `imgsz` | Ukuran masukan | Naikkan bila objek kecil, dengan biaya komputasi |
| Learning rate dan optimizer | Kecepatan dan aturan pembaruan | Mulai dari default, ubah satu per satu |
| `patience` | Early stopping | Hentikan training bila validasi stagnan |
| `seed` | Reproduksibilitas | Catat bersama versi pustaka |

Hasil tersimpan di `runs_ukm/yolo26n_gate_bola/`, termasuk `weights/best.pt` (terbaik pada validasi) dan `weights/last.pt`.

**Reproduksibilitas eksperimen.** Catat: versi Ultralytics dan PyTorch, perangkat, seed, seluruh argumen training, versi dataset, dan hash commit kode. Dua training dengan pengaturan sama dapat tetap berbeda sedikit; ulangi beberapa kali bila selisih antar model kecil.

Bila memori GPU tidak cukup atau tidak ada GPU, jalankan training di Google Colab dengan runtime GPU dan simpan hasil ke penyimpanan permanen.

## 5.7 Evaluasi performa

```python
model = YOLO("runs_ukm/yolo26n_gate_bola/weights/best.pt")
m = model.val(data="dataset/data.yaml", split="val")
print("mAP50-95:", m.box.map, "mAP50:", m.box.map50)
print("precision:", m.box.mp, "recall:", m.box.mr)
print("AP50-95 per kelas:", m.box.maps)
```

Artefak yang dibuat Ultralytics di folder hasil (nama dapat bervariasi antar versi): `results.png` (kurva loss dan metrik), `confusion_matrix.png`, kurva PR/F1, dan contoh prediksi validasi.

Cara membaca:

- **Loss training turun tetapi loss validasi naik** → overfitting. Tambah data, perkuat augmentasi, atau kecilkan model.
- **Metrik rendah di kedua subset** → underfitting atau data/label bermasalah. Periksa kualitas anotasi dan kecukupan data.
- **Confusion matrix** → kelas mana yang tertukar, dan seberapa sering objek terlewat (dianggap latar) atau latar dianggap objek.
- **Metrik per kelas** → kelas yang lemah membutuhkan lebih banyak data atau definisi yang lebih jelas.

Lihat contoh kesalahan secara visual: buka gambar validasi dengan prediksi terburuk dan kelompokkan penyebabnya (objek kecil, pantulan cahaya, motion blur, bola sebagian tertutup, label salah).

## 5.8 Analisis generalisasi

Uji model pada data yang **belum digunakan selama training**, termasuk kondisi yang sengaja dibuat berbeda:

| Kondisi uji | Contoh |
| --- | --- |
| Pencahayaan | Terang, redup, kontras rendah |
| Gerakan robot | Robot diam, robot bergerak (motion blur) |
| Jarak objek | Dekat, sedang, jauh |
| Latar belakang | Lantai arena polos, dinding, objek pengganggu, penonton |

Laporkan metrik per kondisi. Penurunan tajam pada kondisi tertentu memberi tahu data apa yang perlu ditambah.

## 5.9 Praktik utama

Latih YOLO11 dan YOLO26 pada dataset kustom yang sama, dengan **pembagian data dan prosedur evaluasi yang sama**:

```python
for bobot in ["yolo11n.pt", "yolo26n.pt"]:
    nama = bobot.replace(".pt", "")
    YOLO(bobot).train(data="dataset/data.yaml", epochs=100, imgsz=640,
                      batch=16, patience=20, seed=0,
                      project="runs_ukm", name=nama)
```

Bandingkan performa serta kesalahan masing-masing (mAP, precision, recall, confusion matrix, hasil per kondisi 5.8). Catat bahwa satu kali training belum cukup untuk menyimpulkan perbedaan kecil.

## Cek pemahaman

1. Mengapa membagi frame video secara acak ke train dan test menyesatkan?
2. Apa arti loss validasi naik sementara loss training turun?
3. Mengapa gambar tanpa objek (negatif) berguna?
4. Mengapa flip horizontal bisa merusak validitas pada kasus tertentu?

## Kesalahan umum

- Kelas pada label tidak sesuai urutan `names` di `data.yaml`.
- Path di `data.yaml` salah sehingga training berjalan dengan dataset kosong.
- Memakai data uji untuk menyetel hiperparameter.
- Hanya melaporkan mAP keseluruhan tanpa melihat per kelas dan per kondisi.

## Tugas

Serahkan: struktur dataset beserta `data.yaml`, skrip dan log training kedua model, tabel perbandingan metrik, tiga contoh kesalahan terburuk lengkap dengan analisis penyebabnya, dan rencana perbaikan data.
