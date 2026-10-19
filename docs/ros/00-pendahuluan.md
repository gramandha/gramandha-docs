# Modul Pelatihan ROS 2 Jazzy untuk UKM Robotika

## Pendahuluan

Modul ini membawa anggota UKM dari nol sampai mampu menjalankan deteksi objek YOLO di atas ROS 2 Jazzy dalam 7 lesson, masing-masing sekitar 2 jam.

**Tujuan akhir.** Peserta dapat memasang ROS 2, membuat package Python, menulis node publisher dan subscriber, menghubungkan kamera, lalu menjalankan node YOLO yang mempublikasikan hasil deteksi.

**Prasyarat.**

- Laptop dengan Ubuntu 24.04 LTS (instalasi langsung atau dual boot; hindari WSL untuk kamera).
- Ruang disk kosong minimal 20 GB dan RAM minimal 8 GB.
- Pemahaman dasar Python (variabel, fungsi, class) dan terminal Linux (`cd`, `ls`, `mkdir`).
- Kamera USB/webcam untuk Lesson 6 dan 7. GPU NVIDIA bersifat opsional.

**Alur lesson.**

| Lesson | Topik | Hasil yang diverifikasi |
| --- | --- | --- |
| 1 | Instalasi dan pengenalan ROS 2 | `talker` dan `listener` saling terhubung |
| 2 | Workspace dan package | Package `robotik_dasar` berhasil di-build |
| 3 | Node Python | Node sendiri berjalan lewat `ros2 run` |
| 4 | Topic dan message | Publisher dan subscriber `String` berfungsi |
| 5 | Komunikasi antar-node | Laju topic terukur dengan `ros2 topic hz` |
| 6 | Integrasi kamera | Gambar tampil di `rqt_image_view` |
| 7 | Integrasi YOLO | Hasil deteksi terpublikasi sebagai topic |

**Cara memakai modul.** Setiap lesson berisi tujuan, teori singkat, praktik bertahap, cek pemahaman, dan kesalahan umum. Instruktur mendemokan satu kali, lalu peserta mengulang sendiri. Perintah diawali `$` untuk terminal; jangan ikut disalin.
