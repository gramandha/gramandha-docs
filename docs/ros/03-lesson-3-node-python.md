# Lesson 3 — Node Python

Setelah lesson ini, peserta dapat menulis node Python berbasis class `Node`, mendaftarkannya sebagai `console_scripts`, dan menjalankannya dengan `ros2 run`.

**Tujuan belajar.** Peserta mampu (1) menulis class turunan `rclpy.node.Node`, (2) menyusun fungsi `main()` yang benar, (3) mendaftarkan entry point, dan (4) mengelola node dengan siklus hidup yang bersih.

## Teori singkat

Node adalah proses yang melakukan satu tugas tertentu, misalnya membaca kamera atau menghitung deteksi. Di `rclpy`, node ditulis sebagai class yang mewarisi `Node`. Fungsi `main()` selalu mengikuti pola berikut:

1. `rclpy.init()` — menyalakan sistem komunikasi ROS di proses ini.
2. Membuat objek node.
3. `rclpy.spin(node)` — menjaga node tetap hidup dan memproses callback (timer, subscriber).
4. `node.destroy_node()` — membersihkan node setelah `spin` berhenti.
5. `rclpy.shutdown()` — mematikan sistem komunikasi.

## Praktik 3.1 — Node pertama: `halo_node`

Buat file `~/ros2_ws/src/robotik_dasar/robotik_dasar/halo_node.py`:

```python
import rclpy
from rclpy.node import Node


class HaloNode(Node):
    def __init__(self):
        super().__init__('halo_node')
        self.hitungan = 0
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Halo node dimulai')

    def timer_callback(self):
        self.hitungan += 1
        self.get_logger().info(f'Halo UKM Robotika! hitungan = {self.hitungan}')


def main(args=None):
    rclpy.init(args=args)
    node = HaloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Blok `try/except/finally` memastikan node dibersihkan walau pengguna menekan `Ctrl+C`. Pemeriksaan `rclpy.ok()` mencegah error saat `shutdown` sudah dipanggil lebih dulu.

## Praktik 3.2 — Daftarkan `console_scripts`

Buka `setup.py` dan ubah bagian `entry_points`:

```python
entry_points={
    'console_scripts': [
        'halo_node = robotik_dasar.halo_node:main',
    ],
},
```

Formatnya: `nama_perintah = nama_package.nama_file:nama_fungsi`.

Pastikan juga `package.xml` memuat dependensi:

```xml
<exec_depend>rclpy</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

## Praktik 3.3 — Build dan jalankan

```bash
$ cd ~/ros2_ws
$ colcon build --symlink-install --packages-select robotik_dasar
$ source install/setup.bash
$ ros2 run robotik_dasar halo_node
```

Keluaran: log `Halo UKM Robotika!` muncul setiap 1 detik. Hentikan dengan `Ctrl+C`.

## Praktik 3.4 — Latihan

Ubah periode timer menjadi 0,5 detik dan tambahkan parameter `nama` memakai `self.declare_parameter('nama', 'UKM')`. Jalankan dengan `ros2 run robotik_dasar halo_node --ros-args -p nama:=Robotik`.

## Cek pemahaman

1. Apa yang terjadi bila `rclpy.spin()` tidak dipanggil?
2. Mengapa `destroy_node()` dan `shutdown()` ditaruh di blok `finally`?
3. Di bagian mana nama perintah `ros2 run` ditentukan?

## Kesalahan umum

- `No executable found` → entry point belum ditambahkan, atau lupa build ulang dan `source` setelah mengubah `setup.py`.
- Node langsung berhenti → `spin()` tidak dipanggil.
- Nama file dan nama modul di entry point tidak sama.
