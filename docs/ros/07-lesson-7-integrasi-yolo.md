# Lesson 7 — Integrasi YOLO

Setelah lesson ini, peserta dapat menjalankan deteksi objek YOLO pada gambar kamera di ROS 2 dan mempublikasikan hasilnya sebagai topic. Lesson ini tidak menjanjikan angka FPS tertentu, karena performa bergantung pada model, resolusi, perangkat keras, dan pipeline; peserta justru belajar **mengukurnya sendiri**.

**Tujuan belajar.** Peserta mampu (1) menjelaskan arsitektur subscribe–inference–publish, (2) memasang dependensi YOLO secara aman di Ubuntu 24.04, (3) menjalankan paket `yolo_ros` atau node YOLO buatan sendiri, dan (4) mengukur laju keluaran dengan `ros2 topic hz`.

## Teori singkat: arsitektur subscribe–inference–publish

Node YOLO bekerja dalam tiga tahap yang berulang untuk setiap frame:

1. **Subscribe** — menerima `sensor_msgs/msg/Image` dari topic kamera.
2. **Inference** — mengubah pesan menjadi array gambar (lewat `cv_bridge`), lalu menjalankan model YOLO.
3. **Publish** — mengirim hasil: gambar beranotasi dan/atau daftar deteksi (kelas, skor, kotak).

Kamera dan YOLO adalah dua node terpisah yang hanya dihubungkan oleh topic. Karena itu kamera, model, atau tampilan dapat diganti tanpa mengubah bagian lain.

## Faktor yang menentukan performa

| Faktor | Pengaruh |
| --- | --- |
| Ukuran model (misalnya varian kecil vs besar) | Model kecil lebih cepat, akurasi lebih rendah |
| Resolusi masukan | Resolusi lebih tinggi memperlambat inferensi |
| Perangkat keras | CPU saja jauh lebih lambat dibanding GPU |
| Pipeline | Konversi gambar, antrean QoS, dan penggambaran kotak ikut menambah waktu |
| Laju kamera | Inferensi tidak bisa melebihi laju gambar yang masuk |

Karena itu, setiap angka FPS harus dilaporkan bersama konfigurasi ujinya (model, resolusi, perangkat, hasil `ros2 topic hz`).

## Praktik 7.1 — Siapkan lingkungan Python

Ubuntu 24.04 melarang `pip install` global (PEP 668). Gunakan virtual environment yang tetap melihat paket ROS:

```bash
$ sudo apt install python3-venv python3-pip ros-jazzy-cv-bridge -y
$ python3 -m venv --system-site-packages ~/yolo_venv
$ source ~/yolo_venv/bin/activate
$ pip install ultralytics
```

Skrip hasil `ros2 run` memakai Python sistem, sehingga paket di venv perlu ditambahkan ke `PYTHONPATH`:

```bash
$ export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH
```

Paket `ultralytics` menarik PyTorch dan berukuran besar; unduh sebelum hari pelatihan.

## Praktik 7.2 — Opsi A: paket `yolo_ros`

`yolo_ros` adalah paket komunitas yang membungkus YOLO menjadi node ROS 2. Pasang dari sumber ke workspace:

```bash
$ cd ~/ros2_ws/src
$ git clone https://github.com/mgonzs13/yolo_ros.git
$ cd yolo_ros
$ pip install -r requirements.txt
$ cd ~/ros2_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build --symlink-install
```

Catatan penting untuk instruktur: daftar versi dan model YOLO yang didukung, nama launch file, serta nama parameter berubah mengikuti pengembangan paket. Catatan peninjauan yang saya terima menyebut paket ini mencantumkan dukungan YOLOv26, tetapi saya belum memverifikasinya langsung. **Baca README repositori dan uji pada satu mesin sebelum hari pelatihan**, lalu sesuaikan perintah di bawah.

Jalankan mengikuti README, contoh bentuk umumnya:

```bash
$ ros2 launch yolo_bringup yolo.launch.py input_image_topic:=/image_raw
```

Di terminal lain, gunakan alat dari Lesson 5 dan 6:

```bash
$ ros2 topic list -t
$ ros2 topic hz <topic_keluaran_yolo>
$ ros2 run rqt_image_view rqt_image_view
```

## Praktik 7.3 — Opsi B: node YOLO buatan sendiri

Opsi ini lebih mudah dipahami peserta karena seluruh arsitektur terlihat di satu file. Buat `robotik_dasar/yolo_node.py`:

```python
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('conf', 0.5)

        self.bridge = CvBridge()
        self.model = YOLO(self.get_parameter('model').value)
        self.conf = self.get_parameter('conf').value

        self.sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self.on_image, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(Image, 'yolo/annotated', 10)
        self.pub_det = self.create_publisher(String, 'yolo/detections', 10)

    def on_image(self, msg):
        t0 = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hasil = self.model(frame, conf=self.conf, verbose=False)[0]

        nama = [hasil.names[int(c)] for c in hasil.boxes.cls]
        self.pub_det.publish(String(data=', '.join(nama) or 'tidak ada objek'))

        anotasi = hasil.plot()
        out = self.bridge.cv2_to_imgmsg(anotasi, 'bgr8')
        out.header = msg.header
        self.pub_img.publish(out)
        self.get_logger().info(
            f'inferensi {1000*(time.time()-t0):.0f} ms', throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

Daftarkan `'yolo_node = robotik_dasar.yolo_node:main'` di `setup.py`, tambahkan `<exec_depend>sensor_msgs</exec_depend>` dan `<exec_depend>cv_bridge</exec_depend>` di `package.xml`, lalu build ulang.

Catatan: bobot model (misalnya `yolov8n.pt`) diunduh otomatis saat pertama kali dijalankan dan membutuhkan internet. Siapkan file bobot di muka bila jaringan kelas tidak stabil. Pilih nama model sesuai versi `ultralytics` yang terpasang.

## Praktik 7.4 — Jalankan dan ukur

```bash
# Terminal 1
$ ros2 run v4l2_camera v4l2_camera_node
# Terminal 2
$ ros2 run robotik_dasar yolo_node
# Terminal 3
$ ros2 topic hz /image_raw
$ ros2 topic hz /yolo/annotated
$ ros2 topic echo /yolo/detections
$ ros2 run rqt_image_view rqt_image_view   # pilih /yolo/annotated
```

Buat tabel hasil ukur kelompok:

| Model | Resolusi | Perangkat | Laju `/image_raw` (Hz) | Laju `/yolo/annotated` (Hz) |
| --- | --- | --- | --- | --- |
| (isi) | (isi) | CPU / GPU | (isi) | (isi) |

Bandingkan antar kelompok dan diskusikan penyebab perbedaannya.

## Praktik 7.5 — Latihan optimasi

Ubah satu variabel pada satu waktu (model lebih kecil, resolusi kamera lebih rendah, `conf` lebih tinggi) dan catat pengaruhnya pada laju keluaran. Prinsipnya: ubah satu hal, ukur, lalu bandingkan.

## Cek pemahaman

1. Sebutkan tiga tahap pada arsitektur subscribe–inference–publish.
2. Mengapa laju keluaran YOLO tidak bisa melebihi laju kamera?
3. Mengapa klaim "di atas 30 FPS" tidak boleh dijadikan jaminan?

## Kesalahan umum

- `error: externally-managed-environment` saat `pip install` → gunakan virtual environment (Praktik 7.1).
- `ModuleNotFoundError: ultralytics` saat `ros2 run` → `PYTHONPATH` belum menyertakan site-packages venv.
- Laju keluaran jauh di bawah laju kamera → model terlalu besar untuk perangkat; coba model lebih kecil atau resolusi lebih rendah.
- Tidak ada gambar di `rqt_image_view` → periksa nama topic keluaran dan kecocokan QoS (Lesson 4 dan 6).
