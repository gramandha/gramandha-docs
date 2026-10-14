# Lesson 4 — Topic dan message

Setelah lesson ini, peserta dapat membuat publisher dan subscriber `std_msgs/msg/String` dan memahami arti angka `10` pada `create_publisher`, yaitu pengaturan QoS, bukan sekadar jumlah pesan yang dikirim.

**Tujuan belajar.** Peserta mampu (1) menjelaskan model publish–subscribe, (2) menulis publisher dan subscriber, (3) memeriksa tipe pesan, dan (4) menjelaskan dasar QoS.

## Teori singkat

Topic adalah saluran bernama. Node yang mengirim disebut **publisher**, node yang menerima disebut **subscriber**. Setiap topic memiliki satu **tipe message**, misalnya `std_msgs/msg/String` yang hanya berisi satu field `data` bertipe string. Banyak publisher dan subscriber boleh memakai topic yang sama, dan mereka tidak saling mengenal.

## Praktik 4.1 — Periksa tipe message

```bash
$ ros2 interface show std_msgs/msg/String
$ ros2 interface list | grep std_msgs | head
```

## Praktik 4.2 — Publisher

Buat `robotik_dasar/talker.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):
    def __init__(self):
        super().__init__('talker_ukm')
        self.pub = self.create_publisher(String, 'chatter_ukm', 10)
        self.timer = self.create_timer(0.5, self.kirim)
        self.i = 0

    def kirim(self):
        msg = String()
        msg.data = f'Halo dari UKM, pesan ke-{self.i}'
        self.pub.publish(msg)
        self.get_logger().info(f'Kirim: {msg.data}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

## Praktik 4.3 — Subscriber

Buat `robotik_dasar/listener.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self):
        super().__init__('listener_ukm')
        self.sub = self.create_subscription(
            String, 'chatter_ukm', self.terima, 10)

    def terima(self, msg):
        self.get_logger().info(f'Terima: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

Tambahkan kedua entry point di `setup.py`:

```python
'talker_ukm = robotik_dasar.talker:main',
'listener_ukm = robotik_dasar.listener:main',
```

Lalu `colcon build --symlink-install`, `source install/setup.bash`, dan jalankan masing-masing di terminal terpisah.

## Penegasan: arti angka `10` pada QoS

Argumen ketiga `create_publisher(String, 'chatter_ukm', 10)` adalah bentuk ringkas dari profil **QoS (Quality of Service)** dengan `history = KEEP_LAST` dan `depth = 10`. Artinya antrean menyimpan hingga 10 pesan terakhir bila penerima belum sempat memprosesnya. Angka ini bukan jumlah pesan yang dikirim. Nilai lain pada profil tersebut tetap default: `reliability = RELIABLE` dan `durability = VOLATILE`.

Bentuk panjangnya setara:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
self.pub = self.create_publisher(String, 'chatter_ukm', qos)
```

QoS publisher dan subscriber harus **kompatibel**. Misalnya publisher `BEST_EFFORT` tidak cocok dengan subscriber `RELIABLE`, sehingga pesan tidak mengalir tanpa pesan error yang jelas. Periksa dengan:

```bash
$ ros2 topic info /chatter_ukm --verbose
```

Topic sensor seperti kamera lazim memakai profil `SENSOR_DATA` (best effort); ini akan relevan di Lesson 6 dan 7.

## Cek pemahaman

1. Apa isi `std_msgs/msg/String`?
2. Apa arti `depth = 10` pada QoS?
3. Mengapa publisher best effort dan subscriber reliable bisa gagal berkomunikasi?

## Kesalahan umum

- Nama topic berbeda satu huruf sehingga tidak tersambung.
- Tipe message publisher dan subscriber tidak sama.
- Lupa menambahkan `if __name__ == '__main__': main()` pada file yang ingin dijalankan langsung (tidak perlu bila lewat `ros2 run`).
