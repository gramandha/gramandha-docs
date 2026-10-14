# Lesson 6 — Integrasi kamera

Setelah lesson ini, peserta dapat menghubungkan webcam ke ROS 2, memverifikasi topic gambar, dan menampilkannya dengan `rqt_image_view`. Topic `/camera_info` tidak otomatis tersedia pada semua kamera, jadi lesson ini memverifikasi topic gambar lebih dulu.

**Tujuan belajar.** Peserta mampu (1) menjalankan driver kamera, (2) mengenali topic gambar dan tipenya, (3) menampilkan gambar dengan `rqt_image_view`, (4) mengukur laju gambar, dan (5) menjelaskan kapan `CameraInfo` dan `rviz2` dibutuhkan.

## Teori singkat

Driver kamera adalah node yang membaca perangkat (misalnya `/dev/video0`) dan mempublikasikan frame sebagai pesan `sensor_msgs/msg/Image`. Sebagian driver juga mempublikasikan `sensor_msgs/msg/CameraInfo` (parameter kalibrasi lensa), tetapi tidak semua kamera atau konfigurasi melakukannya. Nama topic juga bergantung pada driver dan konfigurasinya, sehingga **jangan mengandalkan nama topic dari tutorial; selalu verifikasi dengan `ros2 topic list`.**

## Praktik 6.1 — Pastikan kamera terbaca Linux

```bash
$ ls /dev/video*
$ sudo apt install v4l-utils -y
$ v4l2-ctl --list-devices
```

Jika kamera tidak muncul, cabut-pasang USB atau coba port lain. Pastikan pengguna ada di grup `video` (`groups`).

## Praktik 6.2 — Pasang dan jalankan driver kamera

Contoh memakai paket `v4l2_camera` (alternatif: `usb_cam`). Nama paket dapat berubah antar rilis, jadi cek ketersediaannya lebih dulu.

```bash
$ apt search ros-jazzy-v4l2-camera
$ sudo apt install ros-jazzy-v4l2-camera -y
$ ros2 run v4l2_camera v4l2_camera_node
```

Untuk memilih kamera lain, tambahkan parameter, misalnya `--ros-args -p video_device:=/dev/video2`.

## Praktik 6.3 — Verifikasi topic gambar

Di terminal lain:

```bash
$ ros2 topic list -t
$ ros2 topic info /image_raw --verbose
$ ros2 topic hz /image_raw
```

Catat tiga hal: (1) apakah topic bertipe `sensor_msgs/msg/Image`, (2) laju aktual (Hz), dan (3) profil QoS publisher (umumnya reliable atau best effort, tergantung driver). Nama `/image_raw` adalah umum, tetapi bukan jaminan.

## Praktik 6.4 — Tampilkan dengan `rqt_image_view`

```bash
$ ros2 run rqt_image_view rqt_image_view
```

Pilih topic gambar pada dropdown di bagian atas jendela. Jika gambar hitam atau tidak muncul, periksa dahulu apakah `ros2 topic hz` melaporkan laju bukan nol, lalu periksa kecocokan QoS.

Untuk awal, `rqt_image_view` adalah alat yang tepat karena hanya membutuhkan topic gambar.

## Praktik 6.5 — Cek `CameraInfo` (opsional)

```bash
$ ros2 topic list | grep -i camera_info
$ ros2 topic echo /camera_info --once
```

Jika topic tidak ada, itu normal untuk driver yang tidak menyediakannya. `CameraInfo` baru wajib bila Anda perlu kalibrasi lensa, perhitungan jarak 3D, atau pemrosesan citra yang memakai parameter intrinsik. Untuk deteksi YOLO 2D di Lesson 7, **tidak diperlukan**.

## Praktik 6.6 — `rviz2` (lanjutan)

```bash
$ rviz2
```

Di RViz, klik **Add → By topic**, lalu pilih tampilan gambar. Berbeda dengan `rqt_image_view`, RViz memerlukan konfigurasi tampilan dan umumnya bergantung pada `CameraInfo` serta pengaturan *frame* untuk display kamera. Gunakan `rviz2` setelah topic gambar terverifikasi dan sebaiknya setelah peserta mengenal konsep frame (TF).

## Cek pemahaman

1. Mengapa nama topic kamera harus diverifikasi, bukan diasumsikan?
2. Kapan `CameraInfo` benar-benar dibutuhkan?
3. Mengapa `rqt_image_view` lebih cocok untuk verifikasi awal dibanding `rviz2`?

## Kesalahan umum

- Gambar tidak muncul padahal topic ada → QoS tidak kompatibel; coba ubah QoS di viewer atau periksa profil publisher.
- Kamera sedang dipakai aplikasi lain (browser, Cheese) → tutup aplikasi tersebut.
- Izin akses `/dev/video0` ditolak → tambahkan pengguna ke grup `video`, lalu login ulang.
- Laju rendah → turunkan resolusi atau frame rate lewat parameter driver.
