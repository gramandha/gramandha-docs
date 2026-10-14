# Evaluasi, proyek mini, dan referensi

Peserta dinyatakan lulus bila mampu membangun dan mendemokan satu sistem lengkap dari kamera sampai hasil deteksi, dengan sistem yang diuji lewat alat command line ROS 2.

## Proyek mini: penghitung objek

Buat node `penghitung` yang subscribe ke `/yolo/detections`, menghitung jumlah deteksi tiap kelas, dan mempublikasikan ringkasan ke topic `/ringkasan` (`std_msgs/msg/String`) satu kali per detik. Syarat penyerahan:

- Package berhasil di-build dari workspace bersih.
- Tiga node (kamera, YOLO, penghitung) berjalan bersamaan.
- Hasil `ros2 node info` dan `ros2 topic hz` untuk setiap node dilampirkan.
- Laporan singkat memuat model, resolusi, perangkat, dan laju terukur.

## Rubrik penilaian

| Aspek | Bobot | Indikator lulus |
| --- | --- | --- |
| Instalasi dan workspace | 15% | `ros2` berjalan, package ter-build |
| Node dan topic | 25% | Publisher dan subscriber berfungsi, entry point benar |
| Integrasi kamera | 20% | Topic gambar terverifikasi dan tampil di `rqt_image_view` |
| Integrasi YOLO | 25% | Hasil deteksi terpublikasi, laju diukur dan dilaporkan |
| Dokumentasi dan presentasi | 15% | Laporan jelas dan demo berjalan |

## Pemecahan masalah cepat

| Gejala | Periksa |
| --- | --- |
| `ros2: command not found` | `source /opt/ros/jazzy/setup.bash` |
| Package tidak ditemukan | `source ~/ros2_ws/install/setup.bash` |
| Node tidak saling melihat | `ROS_DOMAIN_ID`, jaringan, firewall |
| Data tidak mengalir | `ros2 topic info -v` dan kecocokan QoS |
| Gambar kosong | Laju topic dengan `ros2 topic hz`, QoS viewer |
| Perintah baru tidak dikenal | Build ulang setelah ubah `setup.py`, lalu `source` |

## Referensi

- [ROS 2 Jazzy documentation](https://docs.ros.org/en/jazzy/) — tutorial resmi, instalasi, dan konsep.
- [Instalasi ROS 2 Jazzy di Ubuntu (paket deb)](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) — sumber langkah Lesson 1.
- [yolo_ros](https://github.com/mgonzs13/yolo_ros) — repositori paket YOLO untuk ROS 2; periksa README untuk versi model yang didukung.

Tautan di atas belum dibuka ulang pada sesi ini. Cek kembali sebelum pelatihan, karena perintah instalasi dan dukungan paket dapat berubah.
