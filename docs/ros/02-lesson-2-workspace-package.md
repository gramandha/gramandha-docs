# Lesson 2 — Workspace dan package

Setelah lesson ini, peserta memiliki workspace `~/ros2_ws` dan package Python `robotik_dasar` yang berhasil di-build dan dijalankan.

**Tujuan belajar.** Peserta mampu (1) membuat workspace, (2) membuat package `ament_python`, (3) membangun dengan `colcon build --symlink-install`, dan (4) memuat hasil build dengan `source install/setup.bash`.

## Teori singkat

**Workspace** adalah folder tempat kode ROS milik kita disimpan dan dibangun. Struktur standarnya:

```text
ros2_ws/
├── src/        ← kode sumber (package) ditaruh di sini
├── build/      ← file sementara hasil build
├── install/    ← hasil akhir yang dimuat lewat setup.bash
└── log/        ← catatan build
```

**Package** adalah satuan terkecil kode ROS yang dapat dibangun dan dibagikan. Package Python memakai tipe build `ament_python`; package C++ memakai `ament_cmake`. Alat build-nya adalah `colcon`.

## Praktik 2.1 — Pastikan alat build tersedia

Jika `ros-dev-tools` sudah dipasang di Lesson 1, langkah ini hanya verifikasi.

```bash
$ colcon --help
$ sudo apt install python3-colcon-common-extensions python3-rosdep -y
$ sudo rosdep init      # sekali saja per komputer
$ rosdep update
```

Jika `rosdep init` melapor bahwa file sudah ada, abaikan; itu berarti sudah pernah dijalankan.

## Praktik 2.2 — Buat workspace dan package

```bash
$ mkdir -p ~/ros2_ws/src
$ cd ~/ros2_ws/src
$ ros2 pkg create --build-type ament_python --license Apache-2.0 robotik_dasar
```

Perintah ini membuat folder `robotik_dasar` berisi `package.xml`, `setup.py`, `setup.cfg`, dan sub-folder Python `robotik_dasar/`.

## Praktik 2.3 — Build

```bash
$ cd ~/ros2_ws
$ colcon build --symlink-install
```

Opsi `--symlink-install` membuat folder `install` menautkan langsung ke file sumber Python. Akibatnya, perubahan pada skrip Python berlaku tanpa build ulang. Tetapi perubahan pada `setup.py` (misalnya menambah `console_scripts`) tetap memerlukan build ulang.

Build satu package saja:

```bash
$ colcon build --symlink-install --packages-select robotik_dasar
```

## Praktik 2.4 — Muat workspace

```bash
$ source ~/ros2_ws/install/setup.bash
$ ros2 pkg list | grep robotik_dasar
```

Setelah `source`, package muncul di daftar. Aturan urutan: muat ROS (`/opt/ros/jazzy/setup.bash`) lebih dulu, lalu workspace. Tambahkan baris kedua ke `~/.bashrc` bila diinginkan.

## Praktik 2.5 — Kenali isi package

| File | Fungsi |
| --- | --- |
| `package.xml` | Metadata dan dependensi package (`rclpy`, `std_msgs`, dst.) |
| `setup.py` | Cara menginstal package; tempat mendaftarkan `console_scripts` |
| `setup.cfg` | Lokasi instalasi skrip |
| `robotik_dasar/` | Folder kode Python |
| `resource/robotik_dasar` | Penanda package untuk ament |

## Cek pemahaman

1. Apa fungsi folder `src` dan `install`?
2. Kapan build ulang masih diperlukan walau memakai `--symlink-install`?
3. Mengapa terminal baru perlu `source install/setup.bash`?

## Kesalahan umum

- Menjalankan `colcon build` dari dalam `src` → selalu jalankan dari root workspace.
- Package tidak ditemukan setelah build → lupa `source install/setup.bash`.
- Build lama bertabrakan → hapus `build/ install/ log/` lalu build ulang.
