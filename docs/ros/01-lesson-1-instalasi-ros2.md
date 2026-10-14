# Lesson 1 — Instalasi dan pengenalan ROS 2

Setelah lesson ini, peserta memiliki ROS 2 Jazzy yang berjalan di Ubuntu 24.04 dan memahami konsep node, topic, dan message secara garis besar.

**Tujuan belajar.** Peserta mampu (1) memasang `ros-jazzy-desktop`, (2) memuat environment ROS di terminal, (3) menjalankan demo `talker` dan `listener`, dan (4) menjelaskan apa itu node dan topic.

## Teori singkat

ROS 2 (Robot Operating System 2) adalah kerangka kerja middleware untuk membangun perangkat lunak robot. Program dipecah menjadi **node** kecil yang saling bertukar data lewat **topic**. Jazzy Jalisco adalah rilis LTS yang ditargetkan untuk Ubuntu 24.04. Paket `ros-jazzy-desktop` sudah mencakup pustaka inti, RViz, demo, dan tutorial, sehingga tepat untuk pemula.

## Praktik 1.1 — Siapkan locale UTF-8

```bash
$ sudo apt update && sudo apt install locales
$ sudo locale-gen en_US en_US.UTF-8
$ sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
$ export LANG=en_US.UTF-8
```

## Praktik 1.2 — Tambahkan repositori ROS 2

```bash
$ sudo apt install software-properties-common curl -y
$ sudo add-apt-repository universe
$ export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
$ curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
$ sudo dpkg -i /tmp/ros2-apt-source.deb
```

Langkah ini mengikuti dokumentasi resmi. Jika perintah berubah, rujuk halaman instalasi Jazzy (lihat bagian Referensi).

## Praktik 1.3 — Pasang ROS 2 dan alat pengembangan

```bash
$ sudo apt update && sudo apt upgrade -y
$ sudo apt install ros-jazzy-desktop ros-dev-tools -y
```

`ros-dev-tools` memuat `colcon` dan alat build lain yang dibutuhkan di Lesson 2. Unduhan cukup besar, jadi sarankan peserta memakai jaringan yang stabil.

## Praktik 1.4 — Muat environment

```bash
$ source /opt/ros/jazzy/setup.bash
$ echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
$ printenv | grep -i ROS
```

Keluaran harus memuat `ROS_DISTRO=jazzy`. Setiap terminal baru wajib memuat environment ini; baris di `~/.bashrc` melakukannya otomatis.

## Praktik 1.5 — Uji `talker` (C++) dan `listener` (Python)

Terminal 1:

```bash
$ ros2 run demo_nodes_cpp talker
```

Terminal 2:

```bash
$ ros2 run demo_nodes_py listener
```

Terminal 1 mencetak `Publishing: 'Hello World: 1'` dan seterusnya, sedangkan terminal 2 mencetak `I heard: [Hello World: 1]`. Ini membuktikan dua node berbeda bahasa dapat berkomunikasi, dan instalasi berhasil.

## Cek pemahaman

1. Apa beda node dan topic?
2. Mengapa `source /opt/ros/jazzy/setup.bash` harus dijalankan di tiap terminal baru?
3. Mengapa `talker` C++ dapat berbicara dengan `listener` Python?

## Kesalahan umum

- `ros2: command not found` → environment belum di-source.
- Paket tidak ditemukan → repositori `universe` atau ROS belum ditambahkan, atau Ubuntu bukan 24.04.
- Listener tidak menerima pesan → kedua terminal memakai `ROS_DOMAIN_ID` berbeda.
