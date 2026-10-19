# Lesson 5 — Komunikasi antar-node

Setelah lesson ini, peserta dapat menjalankan beberapa node bersamaan, mengamati aliran data, dan mengukur laju topic dengan alat command line ROS 2.

**Tujuan belajar.** Peserta mampu (1) menjalankan dua node di terminal terpisah, (2) memakai `ros2 topic list/echo/hz/info`, (3) memakai `ros2 node list` dan `ros2 node info`, dan (4) memvisualisasikan graf dengan `rqt_graph`.

## Praktik 5.1 — Jalankan dua node di dua terminal

Pastikan setiap terminal sudah memuat ROS dan workspace (`source ~/ros2_ws/install/setup.bash`).

Terminal 1:

```bash
$ ros2 run robotik_dasar talker_ukm
```

Terminal 2:

```bash
$ ros2 run robotik_dasar listener_ukm
```

Keluaran listener harus mengikuti nomor pesan dari talker.

## Praktik 5.2 — Amati dari terminal ketiga

```bash
$ ros2 node list
$ ros2 topic list
$ ros2 topic list -t              # sertai tipe message
$ ros2 topic echo /chatter_ukm    # lihat isi pesan
$ ros2 topic info /chatter_ukm    # jumlah publisher dan subscriber
```

## Praktik 5.3 — Ukur laju dengan `ros2 topic hz`

```bash
$ ros2 topic hz /chatter_ukm
```

Timer di Lesson 4 berperiode 0,5 detik, sehingga laju rata-rata sekitar 2 Hz. Jika tidak, periksa apakah ada beban CPU tinggi atau timer salah diatur. Alat ini akan dipakai lagi di Lesson 6 dan 7 untuk mengukur laju kamera dan keluaran YOLO.

## Praktik 5.4 — Lihat hubungan lewat `ros2 node info`

```bash
$ ros2 node info /talker_ukm
$ ros2 node info /listener_ukm
```

Keluaran mencantumkan, per node: topic yang di-*subscribe*, topic yang di-*publish*, service, dan action. Untuk `/talker_ukm`, daftar *Publishers* memuat `/chatter_ukm: std_msgs/msg/String`. Untuk `/listener_ukm`, topic yang sama muncul di bagian *Subscribers*. Perintah ini adalah cara tercepat memastikan sebuah node benar-benar terhubung ke topic yang dimaksud.

## Praktik 5.5 — Visualisasi graf

```bash
$ rqt_graph
```

Graf menampilkan node sebagai oval dan topic sebagai panah di antaranya. Gunakan tombol refresh setelah menjalankan node baru.

## Praktik 5.6 — Tantangan: dua publisher, satu subscriber

Jalankan satu `listener_ukm` dan dua `talker_ukm` dengan nama node berbeda:

```bash
$ ros2 run robotik_dasar talker_ukm --ros-args -r __node:=talker_a
$ ros2 run robotik_dasar talker_ukm --ros-args -r __node:=talker_b
```

Amati bahwa listener menerima pesan dari keduanya, lalu cek dengan `ros2 topic info /chatter_ukm`.

## Cek pemahaman

1. Perintah apa untuk melihat siapa saja yang publish ke sebuah topic?
2. Apa kegunaan `ros2 node info` dibanding `ros2 topic info`?
3. Mengapa `ros2 topic hz` penting sebelum menjalankan algoritma berat?

## Kesalahan umum

- `ros2 topic hz` tidak menampilkan apa pun → belum ada publisher di topic itu.
- Dua node bernama sama → muncul peringatan; ganti nama lewat `-r __node:=...`.
- Node tidak saling melihat → `ROS_DOMAIN_ID` berbeda atau jaringan memblokir multicast.
