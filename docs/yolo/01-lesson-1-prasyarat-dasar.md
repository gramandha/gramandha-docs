# Lesson 1 — Prasyarat Dasar (Fondasi)

**Tujuan:** Menguasai dasar pemrograman, pengolahan citra, dan deep learning sebagai bekal untuk memahami serta mengimplementasikan YOLO.

**Tujuan belajar.** Peserta mampu (1) menulis Python terstruktur dengan NumPy, (2) memanipulasi citra dengan OpenCV, (3) menjelaskan tensor dan proses training di PyTorch, (4) menjelaskan cara kerja CNN, dan (5) mengubah gambar menjadi tensor yang siap masuk ke model.

## 1.1 Pemrograman Python

Topik yang harus dikuasai: variabel, tipe data, struktur kontrol, fungsi, class, modul, exception handling, virtual environment, dan manipulasi array dengan NumPy. Pandas berguna untuk menganalisis hasil eksperimen tetapi bukan prasyarat utama.

```python
import numpy as np

a = np.arange(12).reshape(3, 4)   # array 3 baris, 4 kolom
print(a.shape, a.dtype)           # (3, 4) int64
print(a[:, 1:3])                  # slicing kolom 1 dan 2
print(a.mean(axis=0))             # rata-rata per kolom
```

Latihan: tulis fungsi `normalisasi(arr)` yang mengubah array ke rentang 0–1 dan menangani kasus array berisi nilai yang sama (pembagian nol).

## 1.2 Computer vision dasar dengan OpenCV

Citra digital adalah array angka. Gambar berwarna berbentuk `(tinggi, lebar, kanal)`. **OpenCV membaca gambar dalam urutan kanal BGR**, sedangkan banyak pustaka lain (Matplotlib, PyTorch dengan model pretrained) mengharapkan RGB. Kesalahan urutan kanal adalah sumber bug yang sangat umum.

```python
import cv2

img = cv2.imread("contoh.jpg")            # BGR, uint8
print(img.shape, img.dtype)               # (H, W, 3) uint8
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kecil = cv2.resize(img, (320, 240))       # (lebar, tinggi)
potong = img[100:300, 200:400]            # [y1:y2, x1:x2]
abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, biner = cv2.threshold(abu, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite("hasil.jpg", kecil)
```

Perhatikan bahwa `cv2.resize` menerima ukuran dalam urutan **(lebar, tinggi)**, sedangkan `img.shape` mengembalikan **(tinggi, lebar)**.

Membaca video dan webcam:

```python
cap = cv2.VideoCapture(0)                 # 0 = kamera pertama, atau path file video
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

## 1.3 Dasar deep learning dengan PyTorch

Konsep inti:

- **Tensor**: array multidimensi yang dapat dijalankan di CPU atau GPU.
- **Forward pass**: data melewati lapisan model sehingga menghasilkan prediksi.
- **Loss function**: ukuran seberapa jauh prediksi dari target.
- **Gradient dan backpropagation**: turunan loss terhadap parameter, dihitung otomatis oleh `autograd`.
- **Optimizer**: aturan memperbarui parameter berdasarkan gradient (misalnya SGD, Adam).

Contoh siklus training minimal:

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(32, 4)
y = x.sum(dim=1, keepdim=True)

for epoch in range(100):
    pred = model(x)            # forward pass
    loss = loss_fn(pred, y)    # hitung loss
    opt.zero_grad()
    loss.backward()            # backpropagation
    opt.step()                 # perbarui parameter
print(loss.item())
```

Pasang PyTorch sesuai instruksi di situs resmi PyTorch (pilih konfigurasi OS, CPU/CUDA Anda). Cek hasilnya:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

## 1.4 Convolutional Neural Network (CNN)

| Konsep | Penjelasan singkat |
| --- | --- |
| Convolution | Filter kecil digeser di atas citra untuk mendeteksi pola lokal (tepi, tekstur) |
| Feature map | Keluaran satu filter; menunjukkan di mana pola tersebut muncul |
| Stride | Jarak geser filter; stride besar menurunkan resolusi keluaran |
| Padding | Tambahan piksel di tepi agar ukuran keluaran dapat dipertahankan |
| Pooling / downsampling | Memperkecil resolusi untuk menghemat komputasi dan memperluas medan pandang |
| Fungsi aktivasi | Memberi non-linearitas, misalnya ReLU atau SiLU |
| Ekstraksi fitur | Lapisan awal menangkap fitur sederhana, lapisan dalam menangkap fitur yang lebih abstrak |

Rumus ukuran keluaran satu sisi: `out = floor((in + 2*padding - kernel) / stride) + 1`.

```python
import torch, torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
x = torch.randn(1, 3, 640, 640)
print(conv(x).shape)     # torch.Size([1, 16, 320, 320])
```

Pada contoh di atas, `floor((640 + 2 - 3) / 2) + 1 = 320`. Cocokkan hasil perhitungan manual dengan keluaran kode.

## 1.5 Praktik utama: dari gambar ke tensor

```python
import cv2
import numpy as np
import torch

img = cv2.imread("contoh.jpg")                    # (H, W, 3) BGR uint8
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (640, 640))
x = torch.from_numpy(rgb).float() / 255.0         # (H, W, 3) float 0-1
x = x.permute(2, 0, 1).unsqueeze(0)               # (1, 3, H, W)

print(x.shape, x.dtype, x.min().item(), x.max().item())
```

Bentuk `(N, C, H, W)` adalah format yang diharapkan model PyTorch untuk citra. Catat setiap langkah transformasi: urutan kanal, rentang nilai, tipe data, dan posisi dimensi.

## Cek pemahaman

1. Mengapa gambar yang dibaca OpenCV tampak berwarna aneh bila ditampilkan langsung di Matplotlib?
2. Apa beda `img.shape` dan argumen `cv2.resize`?
3. Hitung ukuran keluaran konvolusi untuk input 416, kernel 3, stride 1, padding 1.
4. Mengapa `optimizer.zero_grad()` dipanggil sebelum `backward()`?

## Kesalahan umum

- Gambar `None` setelah `cv2.imread` → path salah; OpenCV tidak melempar error.
- Urutan `(lebar, tinggi)` tertukar saat `resize`.
- Lupa mengubah rentang piksel 0–255 menjadi 0–1 sebelum masuk model.
- Dimensi tensor tidak sesuai `(N, C, H, W)`.

## Tugas

Buat skrip `lesson1.py` yang membaca satu gambar, mengubahnya ke tensor `(1, 3, 640, 640)`, melewatkannya melalui dua lapisan `Conv2d` berurutan, dan mencetak bentuk keluaran tiap lapisan beserta perhitungan manualnya.
