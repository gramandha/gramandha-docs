# Lesson 4 — Praktik dan Inferensi Model Pretrained

**Tujuan:** Mampu menjalankan model YOLO pretrained pada gambar, video, dan kamera sebelum melakukan pelatihan dataset kustom.

**Tujuan belajar.** Peserta mampu (1) menyiapkan environment dengan PyTorch dan Ultralytics, (2) menjalankan inferensi pada gambar, video, dan webcam, (3) membaca hasil deteksi secara terprogram, (4) mengukur FPS dan latensi dengan benar, dan (5) memahami keterbatasan model pretrained.

## 4.1 Persiapan environment

```bash
$ source ~/yolo_env/bin/activate
```

Pasang PyTorch lebih dulu sesuai perangkat Anda dengan mengikuti pemilih instruksi di situs resmi PyTorch (pilih OS, pip, dan CPU atau versi CUDA yang sesuai). Setelah itu:

```bash
$ pip install ultralytics
$ yolo checks
```

Perintah `yolo checks` melaporkan versi Python, PyTorch, ketersediaan CUDA, dan perangkat. Verifikasi manual:

```python
import torch, ultralytics
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Bila CUDA tidak tersedia, inferensi tetap berjalan di CPU, hanya lebih lambat. Bobot pretrained diunduh otomatis saat pertama kali dipakai; siapkan koneksi internet atau unduh bobot di muka.

## 4.2 Inferensi pada gambar

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
hasil = model.predict("bus.jpg", conf=0.25, iou=0.7, imgsz=640, save=True)

r = hasil[0]
print("jumlah deteksi:", len(r.boxes))
for kotak in r.boxes:
    kelas = int(kotak.cls)
    print(r.names[kelas],
          f"skor={float(kotak.conf):.2f}",
          "xyxy=", [round(v) for v in kotak.xyxy[0].tolist()])
```

Hasil gambar beranotasi tersimpan di folder `runs/detect/predict*`. Padanan lewat CLI:

```bash
$ yolo predict model=yolo26n.pt source=bus.jpg conf=0.25 save=True
```

Parameter penting:

| Parameter | Fungsi |
| --- | --- |
| `conf` | Confidence threshold |
| `iou` | IoU threshold untuk NMS (relevan pada jalur yang memakai NMS) |
| `imgsz` | Ukuran sisi gambar masukan |
| `classes` | Batasi kelas yang dideteksi, misalnya `classes=[0]` |
| `device` | Pilih perangkat, misalnya `device=0` atau `device="cpu"` |

Pada YOLO26 tersedia pilihan jalur tanpa NMS: `model.predict("bus.jpg", nms=False)`. Bandingkan hasil dan kecepatannya dengan jalur default, dan catat pengaturan yang Anda pakai.

## 4.3 Inferensi video dan webcam

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

for r in model.predict(source=0, stream=True, conf=0.4, verbose=False):
    frame = r.plot()                       # gambar BGR dengan kotak dan label
    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

`source=0` berarti webcam pertama; ganti dengan path video (`"video.mp4"`) untuk file. Opsi `stream=True` memproses frame satu per satu sehingga memori tidak menumpuk.

## 4.4 Mengukur FPS dan latensi dengan benar

Aturan pengukuran:

1. Lakukan **pemanasan** beberapa iterasi (iterasi awal lebih lambat karena inisialisasi).
2. Ukur **banyak** iterasi dan laporkan rata-rata serta sebarannya.
3. Bedakan **latensi inferensi saja** dengan **latensi end-to-end** (baca frame, praproses, inferensi, pascaproses, tampilan).
4. Catat model, resolusi, perangkat, dan pengaturan lain.

```python
import time
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
gambar = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for _ in range(10):                         # pemanasan
    model.predict(gambar, imgsz=640, verbose=False)

waktu = []
for _ in range(100):
    t0 = time.perf_counter()
    model.predict(gambar, imgsz=640, verbose=False)
    waktu.append((time.perf_counter() - t0) * 1000)

waktu = np.array(waktu)
print(f"rata-rata {waktu.mean():.1f} ms, p95 {np.percentile(waktu, 95):.1f} ms, "
      f"FPS ~ {1000 / waktu.mean():.1f}")
```

Gambar acak hanya untuk uji kecepatan; ia tidak merepresentasikan beban deteksi nyata. Untuk angka yang bermakna, ukur pada gambar atau video yang sebenarnya.

## 4.5 Mengamati pengaruh pengaturan

Isi tabel eksperimen satu variabel pada satu waktu:

| Percobaan | Model | imgsz | conf | Jumlah deteksi | Latensi (ms) | Catatan |
| --- | --- | --- | --- | --- | --- | --- |
| A | yolo26n | 640 | 0.25 | | | |
| B | yolo26n | 320 | 0.25 | | | |
| C | yolo26n | 640 | 0.60 | | | |
| D | yolo26s | 640 | 0.25 | | | |

Pertanyaan analisis: bagaimana resolusi memengaruhi objek kecil? Apa yang terjadi pada jumlah deteksi saat `conf` dinaikkan?

## 4.6 Eksplorasi multi-task

Checkpoint pretrained berbeda untuk tugas berbeda. Contoh penamaan pada keluarga YOLO26 (sesuai dokumentasi, cek ketersediaan pada versi Ultralytics Anda):

| Tugas | Contoh nama file |
| --- | --- |
| Deteksi | `yolo26n.pt` |
| Segmentasi instans | `yolo26n-seg.pt` |
| Pose | `yolo26n-pose.pt` |
| Klasifikasi | `yolo26n-cls.pt` |
| Oriented bounding box | `yolo26n-obb.pt` |

```python
from ultralytics import YOLO
seg = YOLO("yolo26n-seg.pt")
seg.predict("bus.jpg", save=True)
```

## 4.7 Keterbatasan model pretrained

Model pretrained pada COCO mengenal 80 kelas umum (orang, mobil, botol, dan seterusnya). Model itu **tidak otomatis** mampu mendeteksi kelas khusus yang belum pernah dipelajarinya, misalnya gate (gerbang) pada arena, bola dengan warna dan ukuran tertentu, atau komponen robot Anda. Untuk itu diperlukan pelatihan dataset kustom (Lesson 5).

## 4.8 Praktik utama

Jalankan model deteksi YOLO11 dan YOLO26 pada gambar yang sama (minimal 5 gambar beragam). Bandingkan:

1. Hasil visual: objek yang terdeteksi, terlewat, dan salah.
2. Waktu inferensi (dengan prosedur di 4.4).
3. Kebutuhan komputasi (`model.info()`, penggunaan memori).

Tulis kesimpulan beserta batasannya (jumlah gambar kecil, bukan evaluasi akurasi).

## Cek pemahaman

1. Mengapa iterasi pertama inferensi lebih lambat?
2. Apa beda latensi inferensi dan latensi end-to-end?
3. Mengapa model COCO tidak dapat langsung mendeteksi gate pada arena robot?
4. Apa fungsi `stream=True` pada video?

## Kesalahan umum

- `ModuleNotFoundError` → environment belum aktif atau paket belum terpasang.
- Mengukur FPS tanpa pemanasan atau hanya dari satu frame.
- Membandingkan model pada resolusi atau perangkat berbeda lalu menyimpulkan satu lebih baik.
- Webcam tidak terbaca → indeks kamera salah atau dipakai aplikasi lain.

## Tugas

Kumpulkan laporan singkat: tabel eksperimen 4.5, perbandingan YOLO11 vs YOLO26 pada 5 gambar, dan satu contoh kegagalan model pretrained beserta dugaan penyebabnya.
