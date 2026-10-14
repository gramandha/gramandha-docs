# Lesson 6 — Optimasi dan Deployment

**Tujuan:** Mampu mengoptimalkan model dan menjalankannya pada perangkat target dengan mempertimbangkan akurasi, kecepatan, penggunaan memori, dan kebutuhan sistem.

**Tujuan belajar.** Peserta mampu (1) mengekspor model ke format yang sesuai perangkat, (2) memvalidasi kesetaraan hasil ekspor, (3) menerapkan dan mengukur optimasi (FP16, INT8, resolusi, batch), (4) melakukan benchmarking end-to-end, (5) membuat API bila diperlukan, (6) menjalankan model di perangkat edge, dan (7) mengintegrasikan YOLO ke ROS 2.

> **Prinsip.** Jangan mengasumsikan semua format, kuantisasi, atau operator tersedia di semua model, runtime, dan NPU. Setiap langkah optimasi harus diikuti pengukuran akurasi dan latensi.

## 6.1 Export model

Ultralytics menyediakan `model.export()`. Nama format dan opsi dapat berubah antar versi; cek halaman Export pada dokumentasi resmi untuk daftar yang berlaku pada versi Anda.

```python
from ultralytics import YOLO

model = YOLO("runs_ukm/yolo26n_gate_bola/weights/best.pt")

model.export(format="onnx", imgsz=640)                 # ONNX
# model.export(format="engine", imgsz=640, half=True)  # TensorRT (perlu GPU NVIDIA dan TensorRT)
# model.export(format="tflite", imgsz=640)             # LiteRT/TFLite, bila didukung
```

Padanan CLI: `yolo export model=best.pt format=onnx imgsz=640`.

| Format | Cocok untuk | Catatan |
| --- | --- | --- |
| PyTorch (`.pt`) | Pengembangan, training | Referensi akurasi |
| ONNX | Portabel antar runtime (ONNX Runtime, dsb.) | Titik awal ekspor yang umum |
| TensorRT | GPU NVIDIA, termasuk Jetson | Dibangun per perangkat dan versi TensorRT |
| LiteRT / TFLite | Perangkat mobile dan sebagian edge | Dukungan operator terbatas, periksa kompatibilitas |
| OpenVINO | CPU/GPU Intel | Alternatif untuk perangkat Intel |

Pada YOLO26, opsi `nms=False` saat ekspor menghasilkan model end-to-end tanpa NMS yang menyederhanakan pascaproses di luar model (lihat dokumentasi End-to-End Detection Ultralytics). Pastikan kode pascaproses Anda cocok dengan format keluarannya: jalur NMS-free menghasilkan keluaran per deteksi (maks. 300), sedangkan jalur default menghasilkan keluaran mentah yang membutuhkan NMS.

## 6.2 Validasi hasil ekspor

Model hasil ekspor harus menghasilkan prediksi yang setara dengan model asal.

```python
from ultralytics import YOLO

asal = YOLO("best.pt")
ekspor = YOLO("best.onnx")

for gambar in ["uji1.jpg", "uji2.jpg", "uji3.jpg"]:
    a = asal.predict(gambar, conf=0.25, verbose=False)[0]
    b = ekspor.predict(gambar, conf=0.25, verbose=False)[0]
    print(gambar, "jumlah deteksi:", len(a.boxes), "vs", len(b.boxes))
```

Validasi kuantitatif pada dataset yang sama:

```python
m_asal = YOLO("best.pt").val(data="dataset/data.yaml")
m_onnx = YOLO("best.onnx").val(data="dataset/data.yaml")
print("mAP50-95 asal:", m_asal.box.map, " ONNX:", m_onnx.box.map)
```

Perbedaan kecil wajar karena presisi numerik. Selisih besar menandakan masalah ekspor, praproses, atau pascaproses yang tidak cocok.

## 6.3 Optimasi inferensi

| Teknik | Efek | Risiko |
| --- | --- | --- |
| **FP16** (half precision) | Lebih cepat dan hemat memori di perangkat yang mendukung | Perubahan akurasi kecil; tidak semua CPU diuntungkan |
| **INT8 quantization** | Ukuran lebih kecil dan umumnya lebih cepat di perangkat yang mendukung | Akurasi dapat turun; perlu data kalibrasi yang representatif; dukungan bergantung pada model, runtime, dan NPU |
| **Batch size** | Menaikkan throughput | Menaikkan latensi per gambar; tidak cocok untuk aplikasi waktu nyata satu kamera |
| **Resolusi input** | Resolusi lebih rendah mempercepat | Objek kecil dapat terlewat |
| **Ukuran model** | Model lebih kecil lebih cepat | Akurasi lebih rendah |
| **Runtime** | Runtime khusus perangkat (TensorRT, OpenVINO) sering lebih cepat | Perlu ekspor ulang per perangkat |

Contoh ekspor dengan opsi presisi (cek dukungan pada versi dan format Anda):

```python
model.export(format="engine", half=True)                       # FP16 di TensorRT
model.export(format="engine", int8=True, data="dataset/data.yaml")  # INT8 dengan kalibrasi
```

Data kalibrasi INT8 harus mewakili kondisi operasional. Evaluasi ulang mAP setelah kuantisasi.

## 6.4 Benchmarking

Ukur dalam kondisi yang sama untuk tiap konfigurasi:

| Metrik | Cara mengukur |
| --- | --- |
| Latensi end-to-end | Dari frame masuk sampai hasil siap dipakai (praproses + inferensi + pascaproses) |
| Throughput / FPS | Jumlah frame diproses per detik pada beban berkelanjutan |
| RAM / VRAM | `nvidia-smi`, `htop`, atau `tegrastats` pada Jetson |
| Konsumsi daya | Pengukur daya eksternal atau alat bawaan perangkat bila tersedia |
| Akurasi setelah optimasi | `val()` pada dataset yang sama |

```python
import time, numpy as np
from ultralytics import YOLO

def benchmark(path, gambar, n=100, imgsz=640):
    m = YOLO(path)
    for _ in range(10):
        m.predict(gambar, imgsz=imgsz, verbose=False)      # pemanasan
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.predict(gambar, imgsz=imgsz, verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return t.mean(), np.percentile(t, 95)

gambar = "uji1.jpg"
for p in ["best.pt", "best.onnx"]:
    rata, p95 = benchmark(p, gambar)
    print(f"{p}: rata-rata {rata:.1f} ms, p95 {p95:.1f} ms")
```

Laporkan hasil bersama konfigurasi lengkapnya:

| Format | Presisi | imgsz | Perangkat | Latensi rata-rata (ms) | p95 (ms) | mAP50-95 | RAM/VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch | FP32 | 640 | | | | | |
| ONNX | FP32 | 640 | | | | | |
| TensorRT | FP16 | 640 | | | | | |

## 6.5 API development (opsional)

API REST adalah opsi deployment bila model perlu diakses oleh aplikasi lain. API **bukan** prasyarat untuk menjalankan YOLO atau ROS 2.

```python
# api.py
import io
import numpy as np
from fastapi import FastAPI, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best.pt")

@app.post("/deteksi")
async def deteksi(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = model.predict(np.array(img), conf=0.25, verbose=False)[0]
    return [
        {"kelas": r.names[int(b.cls)],
         "skor": float(b.conf),
         "xyxy": [float(v) for v in b.xyxy[0]]}
        for b in r.boxes
    ]
```

Jalankan dengan `pip install fastapi uvicorn python-multipart` lalu `uvicorn api:app --host 0.0.0.0 --port 8000`. Uji: `curl -F file=@uji1.jpg http://localhost:8000/deteksi`. Untuk penggunaan nyata tambahkan autentikasi, batas ukuran unggahan, dan penanganan kesalahan.

## 6.6 Edge AI

Perangkat yang umum: NVIDIA Jetson, Raspberry Pi, atau perangkat dengan NPU yang memiliki runtime kompatibel.

Langkah kerja yang aman:

1. Tentukan target: latensi maksimum yang dapat diterima, batas daya, dan batas memori.
2. Periksa dukungan perangkat: versi sistem operasi/JetPack, runtime (TensorRT, ONNX Runtime, runtime NPU), dan dukungan operator untuk model Anda. Rujuk dokumentasi resmi vendor dan panduan integrasi Ultralytics untuk perangkat tersebut.
3. Ekspor model **di atau untuk** perangkat target (khusus TensorRT, mesin dibangun untuk perangkat dan versi tertentu).
4. Validasi akurasi di perangkat, bukan hanya di laptop.
5. Ukur latensi, FPS, RAM, suhu, dan daya pada beban berkelanjutan; perangkat kecil dapat menurunkan kinerja bila panas (*thermal throttling*).
6. Pilih model terkecil yang memenuhi target akurasi.

## 6.7 Integrasi ROS 2

Arsitektur: node kamera mempublikasikan `sensor_msgs/msg/Image`; node YOLO melakukan *subscribe*, inferensi, lalu *publish* hasil; node lain memakai hasil tersebut untuk visualisasi atau logika kontrol.

```text
Kamera robot → /image_raw → [node YOLO] → /yolo/detections → [node kontrol / visualisasi]
                                      → /yolo/annotated
```

Rujuk modul ROS 2 Jazzy (Lesson 6 dan 7) untuk dasar node dan kamera. Versi yang diperluas dengan pesan terstruktur:

```python
import json, time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('model', 'best.pt')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('imgsz', 640)

        self.bridge = CvBridge()
        self.model = YOLO(self.get_parameter('model').value)
        self.conf = self.get_parameter('conf').value
        self.imgsz = self.get_parameter('imgsz').value

        self.sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self.on_image, qos_profile_sensor_data)
        self.pub_det = self.create_publisher(String, 'yolo/detections', 10)
        self.pub_img = self.create_publisher(Image, 'yolo/annotated', 10)

    def on_image(self, msg):
        t0 = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               verbose=False)[0]

        deteksi = [{
            'kelas': r.names[int(b.cls)],
            'skor': round(float(b.conf), 3),
            'xyxy': [round(float(v), 1) for v in b.xyxy[0]],
        } for b in r.boxes]
        self.pub_det.publish(String(data=json.dumps({
            'stamp': {'sec': msg.header.stamp.sec,
                      'nanosec': msg.header.stamp.nanosec},
            'deteksi': deteksi,
        })))

        out = self.bridge.cv2_to_imgmsg(r.plot(), 'bgr8')
        out.header = msg.header
        self.pub_img.publish(out)

        self.get_logger().info(
            f'{len(deteksi)} objek, {1000*(time.perf_counter()-t0):.0f} ms',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

Catatan desain:

- Model dapat berupa `.pt` atau hasil ekspor (misalnya `.onnx` atau `.engine`) yang dimuat lewat `YOLO(path)`.
- Header pesan hasil disalin dari gambar masukan agar penerima dapat menyinkronkan hasil dengan frame asalnya.
- JSON dalam `std_msgs/String` cocok untuk belajar. Untuk sistem produksi, pertimbangkan pesan terstruktur (misalnya `vision_msgs/Detection2DArray`) agar tipe datanya jelas.
- Pada Ubuntu 24.04, pasang dependensi Python di virtual environment yang melihat paket sistem dan atur `PYTHONPATH` seperti pada modul ROS 2 Lesson 7.

## 6.8 Pengujian sistem

Ukur perilaku sistem utuh dalam kondisi operasional:

| Aspek | Cara |
| --- | --- |
| Kestabilan aliran gambar | `ros2 topic hz /image_raw` selama beberapa menit; catat penurunan |
| Frekuensi deteksi | `ros2 topic hz /yolo/detections` dan bandingkan dengan laju kamera |
| Latensi komunikasi | Bandingkan stempel waktu header gambar dengan waktu penerimaan hasil |
| Kebutuhan sumber daya | `htop`, `nvidia-smi` atau `tegrastats` saat sistem berjalan |
| Ketahanan | Uji kamera terputus, beban CPU tinggi, dan restart node |
| Akurasi lapangan | Cocokkan deteksi dengan pengamatan manusia pada rekaman uji |

Jika frekuensi keluaran jauh di bawah laju kamera, pertimbangkan model lebih kecil, resolusi lebih rendah, format yang dioptimalkan, atau melewatkan frame (*frame skipping*) dengan sadar dan terukur.

## 6.9 Praktik utama: deteksi gate dan bola pada robot

1. Ekspor model terbaik dari Lesson 5 ke ONNX atau format yang sesuai perangkat.
2. Uji kesetaraan hasil terhadap model `.pt` (6.2).
3. Jalankan benchmark (6.4) pada perangkat target.
4. Integrasikan dengan ROS 2 Jazzy memakai node 6.7 untuk mendeteksi gate dan bola.
5. Lakukan pengujian sistem (6.8) dengan rekaman atau uji di arena.
6. Tulis laporan: konfigurasi akhir, metrik akurasi, latensi, keterbatasan, dan rekomendasi.

## Cek pemahaman

1. Mengapa model hasil ekspor harus divalidasi terhadap model asal?
2. Mengapa INT8 membutuhkan data kalibrasi yang representatif?
3. Mengapa batch besar tidak selalu cocok untuk aplikasi waktu nyata?
4. Apa keuntungan menyalin header pesan gambar ke pesan hasil?

## Kesalahan umum

- Mengekspor di satu mesin lalu menjalankan mesin TensorRT di perangkat berbeda.
- Mengukur FPS hanya pada model, bukan pada pipeline end-to-end.
- Format keluaran model (dengan atau tanpa NMS) tidak cocok dengan kode pascaproses.
- Memakai gambar uji yang tidak mewakili kondisi lapangan untuk kalibrasi atau validasi.

## Tugas akhir

Serahkan paket ROS 2 yang berisi node deteksi, file peluncuran (*launch*), model terekspor, dan laporan benchmark (tabel 6.4) beserta bukti pengujian sistem (6.8).
