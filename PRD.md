# PRD: Analisis Ketahanan FCOS vs RetinaNet terhadap Serangan AFOG dan AFOG-V pada Deteksi Mahasiswa Awake/Drowsy

| Item | Isi |
|---|---|
| Versi | 1.4 (diselaraskan dengan proposal penelitian) |
| Platform | Kaggle (format `.ipynb`), 2 akun |
| Bahasa/Framework | Python, PyTorch, Torchvision |
| Status | Draft |

> Keterangan: nilai bertanda **(usulan)** belum ada di dokumen metodologi dan perlu dikonfirmasi sebelum eksekusi.

---

## 1. Latar Belakang dan Tujuan

Deteksi kondisi mahasiswa (awake/drowsy) berbasis object detection rentan terhadap serangan adversarial visual. Penelitian ini membandingkan ketahanan dua filosofi arsitektur:

- **FCOS** (anchor-free)
- **RetinaNet** (anchor-based)

keduanya dengan backbone ResNet50+FPN, terhadap serangan **AFOG** dan **AFOG-V**.

### Tujuan
1. Melatih FCOS dan RetinaNet dengan konfigurasi identik untuk mendeteksi kelas `awake` dan `drowsy`.
2. Mengukur kinerja baseline pada citra bersih.
3. Mengukur degradasi kinerja setelah serangan AFOG dan AFOG-V (ε = 8/255).
4. Menyimpulkan arsitektur mana yang paling tangguh (robust).

### Di Luar Cakupan
- Serangan di luar AFOG/AFOG-V, defense/adversarial training, variasi ε (opsional bila waktu cukup).
- Serangan black-box dan targeted.

---

## 2. Dataset

| Parameter | Nilai |
|---|---|
| Format anotasi | COCO (`bbox` = `[x, y, width, height]` piksel pada ruang 640 × 480) |
| Kelas | 2: `awake`, `drowsy` |
| Resolusi citra | Native **640 × 480 piksel** untuk semua split (tanpa perlu rescale) |
| Bounding box per gambar | **1 bbox wajah** per gambar (14.000 anotasi untuk 14.000 gambar train) |
| Total data | 17.500 gambar, split 80:10:10, seimbang antara awake dan drowsy |
| Jumlah data training | 14.000 gambar (7.000 awake + 7.000 drowsy) |
| Split | **Sudah tersedia** (train / valid / test). Tiap split punya satu `annotation.coco.json` dan folder `images/` yang dipisah ke subfolder `awake/` dan `drowsy/` |
| Jumlah data validasi | 1.750 gambar (875 awake + 875 drowsy) |
| Jumlah data uji | 1.750 gambar (875 awake + 875 drowsy) |
| Sumber di Kaggle | `/kaggle/input/nama-dataset/` (read-only; slug asli disesuaikan, cek dengan `!ls /kaggle/input`) |

### Struktur Direktori Dataset

```
/kaggle/input/nama-dataset/
├── train/
│   ├── annotation.coco.json
│   └── images/
│       ├── awake/
│       └── drowsy/
├── valid/
│   ├── annotation.coco.json
│   └── images/
│       ├── awake/
│       └── drowsy/
└── test/
    ├── annotation.coco.json
    └── images/
        ├── awake/
        └── drowsy/
```

**Ketentuan struktur:**
- Satu file anotasi per split mencakup **semua** gambar split tersebut (awake + drowsy), bukan satu file per kelas.
- Gambar tersebar di dua subfolder kelas, sehingga path gambar **tidak** cukup `images/<file_name>`. Field `file_name` di JSON bisa berupa nama file saja atau path relatif (`awake/xxx.jpg`). Sesuai notebook 01a/01b: dataset class membangun indeks `basename -> path lengkap` sekali di awal dengan `os.walk` atas `images/` (mencakup `awake/` dan `drowsy/`, ekstensi `.jpg/.jpeg/.png/.bmp/.webp`), lalu meresolusi setiap `file_name` lewat `Path(file_name).name`. Hentikan dengan error jika ada nama file ganda dalam satu split atau ada `file_name` yang tidak ditemukan.
- **Label kelas diambil dari `category_id` di anotasi**, bukan dari nama subfolder. Subfolder hanya dipakai sebagai pemeriksaan konsistensi (notebook 00): kategori objek pada gambar di `awake/` harus `awake`, dan sebaliknya.
- Pemetaan `category_id` COCO ke label Torchvision dibuat eksplisit (label 0 dicadangkan untuk background; `awake` dan `drowsy` mendapat label 1 dan 2). Jika JSON memuat kategori tambahan (misalnya supercategory dengan id 0), kategori itu diabaikan dan tidak dihitung sebagai kelas.

**Resolusi citra dan ruang koordinat bounding box (hasil verifikasi `annotation.coco.json`):**

- Semua gambar di `train`, `valid`, dan `test` **sudah beresolusi native 640 × 480 piksel** (resolusi video NTHU-DDD). Field `width` dan `height` di JSON bernilai 640 × 480 untuk seluruh 17.500 gambar, dan file gambar fisik juga 640 × 480. Karena itu resize ke 640 × 480 bersifat *identity*: tidak ada perubahan rasio aspek dan tidak ada distorsi.
- `bbox` COCO berformat `[x, y, width, height]` dalam **piksel pada ruang 640 × 480** — bukan koordinat ternormalisasi dan bukan koordinat resolusi lain. Terverifikasi pada `train`: nilai `x + width` maksimum 547,98 dan `y + height` maksimum 480,0, seluruh bbox di dalam batas citra, tidak ada bbox degenerat (lebar/tinggi 0), dan `area` sama dengan `width × height`.
- Konsekuensinya, pada resize ke 640 × 480 faktor skala bounding box adalah `sx = 640 / 640 = 1.0` dan `sy = 480 / 480 = 1.0`; koordinat bbox tidak bergeser. Notebook tetap menghitung `sx`/`sy` dari `width`/`height` asli agar aman bila kelak dipakai citra berukuran lain.
- Hasil verifikasi `train`: 14.000 gambar dan 14.000 anotasi (tepat **1 bbox per gambar**), `image_id` unik, `file_name` unik, tidak ada gambar tanpa anotasi, dan `category_id` konsisten dengan subfolder (`awake` = 1, `drowsy` = 2). Uji silang dengan Haar cascade frontal-face pada 80 gambar memberi IoU rata-rata 0,66 terhadap bbox GT, sehingga bbox memang menempel pada wajah dan bukan koordinat yang salah skala.
- Catatan penting: jangan mengubah `bbox` memakai skala lain (misalnya membagi dengan 640/480) karena akan menggeser kotak. Satu-satunya transformasi bbox adalah konversi ke `[x1, y1, x2, y2]` untuk Torchvision.

**Ketentuan:**
- Split sudah disediakan, jadi **tidak dibuat ulang**. Dataset yang sama persis dipakai di kedua akun (share, bukan upload ulang berbeda).
- Notebook 00 memverifikasi **kebocoran data** antar split (gambar duplikat/hash sama di train vs valid/test), karena split acak per frame dapat membuat wajah yang sama muncul di train dan test.
- Urutan gambar uji dan seed harus sama di kedua akun.

---

## 3. Konfigurasi Pelatihan (Tabel 3.1)

Seluruh parameter **identik** untuk kedua model.

| Parameter | Nilai |
|---|---|
| Struktur input (`CFG`) | `DATASET_ROOT` = root dataset di `/kaggle/input/...` (slug disesuaikan per akun), `TRAIN_DIR="train"`, `VALID_DIR="valid"`, `ANNOTATION_NAME="annotation.coco.json"` |
| Resolusi input | **640 × 480 piksel (lebar × tinggi)** — `IMAGE_WIDTH=640`, `IMAGE_HEIGHT=480`, sesuai `CFG` notebook 01a. Dataset sudah native 640 × 480 (lihat Bagian 2), jadi resize bilinear langsung ke 640 × 480 bersifat *identity* dan bbox tidak bergeser (`sx = sy = 1.0`). Kode tetap menghitung `sx = 640 / lebar_asli`, `sy = 480 / tinggi_asli` lalu meng-clip bbox ke batas citra, agar aman untuk citra berukuran lain. Transform internal model juga dikunci untuk kedua model: `model.transform.min_size = (480,)`, `max_size = 640` (tidak melakukan penskalaan ulang untuk input 640 × 480; penting untuk RetinaNet yang default-nya `min_size=800`/`max_size=1333`) |
| Batch size | **8** (micro batch 8, gradient accumulation 1; jika OOM: micro batch 4 + gradient accumulation 2, efektif tetap 8) |
| Learning rate awal | **0.0001** |
| Optimizer | **AdamW** |
| Jumlah epoch | **70** (maksimum) |
| Preprocessing bounding box | Konversi COCO ke `[x1, y1, x2, y2]` Torchvision, `area = (x2-x1)*(y2-y1)`, `iscrowd` dipertahankan, bbox degenerat (luas 0) dibuang |
| Bobot awal FCOS | `fcos_resnet50_fpn` (pre-trained) |
| Bobot awal RetinaNet | `retinanet_resnet50_fpn_v2` (pre-trained) |
| Backbone | ResNet50 + FPN (kedua model) |
| Jumlah kelas head | **3 output** (0 background, 1 awake, 2 drowsy) sesuai konvensi Torchvision; layer `cls_logits` FCOS diganti dan diinisialisasi ulang, bobot pretrained lain dipertahankan |

### Tambahan Operasional (usulan, tidak mengubah Tabel 3.1)

| Item | Nilai |
|---|---|
| Sumber kebenaran konfigurasi | Blok `CFG` + sel SETUP di `01a-training-fcos.ipynb` / `01b-training-retinanet.ipynb`. Jika dokumen ini berbeda dari notebook, nilai di notebook yang dipakai (dan PRD direvisi) |
| Mixed precision (AMP) | Aktif (`CFG["AMP"]=True`), `torch.autocast(device_type="cuda", dtype=torch.float16)` + `GradScaler`, hanya bila CUDA tersedia |
| Weight decay | **0.0001** (argumen `weight_decay` pada AdamW) |
| LR scheduler | `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)` yang di-`step` dengan **mAP@0.5 validasi** setiap epoch |
| Seleksi model terbaik | Berdasarkan **mAP@0.5 validasi**, bukan epoch terakhir |
| Early stopping | Patience 10 epoch pada mAP validasi (maks. tetap 70) |
| Checkpoint | Tiap epoch: model, optimizer, scheduler, GradScaler, epoch, `best_map`, history, `epochs_without_improvement`, `CFG` + hash, `SETUP_VERSION`, seed, nama GPU; `best.pth` hanya ditulis saat mAP membaik |
| Lokasi artefak | Checkpoint: `/kaggle/working/checkpoints_fcos`; log + `.pth` final: `/kaggle/working/fcos_training_outputs` (RetinaNet: `..._retinanet`, `retinanet_training_outputs`) |
| Resume | Wajib, karena sesi Kaggle maksimal sekitar 12 jam; isi `RESUME_CHECKPOINT` dengan `fcos_last.pth` / `retinanet_last.pth` |
| `num_workers` | **2** (`persistent_workers=True` bila > 0) |
| Seed | **42** untuk semua notebook (`random`, NumPy, PyTorch, `PYTHONHASHSEED`, cuDNN deterministik) |
| Ambang evaluasi | Confidence 0.5, IoU 0.5, score floor 0.05 saat mengumpulkan prediksi validasi |
| Bobot pratatih | `USE_COCO_PRETRAINED=True` → `FCOS_ResNet50_FPN_Weights.DEFAULT` / `RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT`; bila `False` → backbone `ResNet50_Weights.IMAGENET1K_V2` tanpa bobot deteksi |
| Normalisasi piksel | Dataset hanya `TF.to_tensor` (skala [0,1]); normalisasi ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) dilakukan internal oleh transform model |
| DataLoader | Train `shuffle=True`, valid `shuffle=False`, `drop_last=False`, `pin_memory` bila CUDA, `collate_fn` mengembalikan tuple `(images, targets)` |
| Pengaman loop | Loss non-finite → `RuntimeError`; backward dibagi `GRAD_ACCUM_STEPS`; log progress tiap 50 iterasi |

### Estimasi Beban Training
- 14.000 / 8 = **1.750 iterasi per epoch**, 70 epoch = **122.500 iterasi per model**.
- Perkiraan 20-30 jam per model dengan AMP (ukur nyata dari 50 iterasi pertama), sekitar 2-3 sesi per akun.

---

## 4. Skenario Pengujian

### 4.1 Skenario Bersih (Tabel 3.2): Baseline

| ID | Model | Jenis Citra | Konfigurasi Serangan | Metrik |
|---|---|---|---|---|
| S1-A | FCOS | Bersih | Tanpa injeksi (baseline) | mAP@0.5, Precision, Recall |
| S2-A | RetinaNet | Bersih | Tanpa injeksi (baseline) | mAP@0.5, Precision, Recall |

Metrik tambahan yang direkam: total objek terdeteksi, rata-rata confidence score, FPS inferensi.

### 4.2 Konfigurasi Serangan

| Parameter | AFOG | AFOG-V |
|---|---|---|
| Skenario | White-box, untargeted | White-box, untargeted |
| Sasaran loss | Galat klasifikasi dan regresi bounding box secara umum | Menekan skor kepercayaan objek di bawah ambang deteksi (vanishing) |
| Mekanisme | Attention map + gradien loss, iteratif | Sama |
| Epsilon (ε) | **8/255** | **8/255** |
| Jumlah iterasi | **10** (usulan, cocokkan dengan paper/kode AFOG) | sama dengan AFOG |
| Step size | **2/255** (usulan; total gerak 10 × 2/255 melebihi ε, sehingga proyeksi ke ε-ball aktif) | sama dengan AFOG |
| Norma | L∞ | L∞ |
| Kuantisasi | Bulatkan ke uint8 sebelum inferensi (`round(x*255)/255`) | sama |

Semua parameter serangan **identik** untuk FCOS dan RetinaNet. Perturbasi dibangkitkan dari model yang diserang (white-box), sehingga citra adversarial FCOS hanya dievaluasi pada FCOS, dan sebaliknya.

### 4.3 Skenario Adversarial (Tabel 3.3)

| ID | Model | Jenis Citra | Konfigurasi Serangan | Metrik |
|---|---|---|---|---|
| S1-B | FCOS | Adversarial | AFOG (ε = 8/255) | ΔmAP, ASR |
| S2-B | RetinaNet | Adversarial | AFOG (ε = 8/255) | ΔmAP, ASR |
| S1-C | FCOS | Adversarial | AFOG-V (ε = 8/255) | ΔmAP, ASR |
| S2-C | RetinaNet | Adversarial | AFOG-V (ε = 8/255) | ΔmAP, ASR |

Metrik tambahan: jumlah objek terdeteksi, rata-rata confidence score pada bounding box yang bertahan, jumlah false negative.

### 4.4 Ringkasan 6 Skenario

| Model | Bersih | AFOG | AFOG-V |
|---|---|---|---|
| FCOS | S1-A | S1-B | S1-C |
| RetinaNet | S2-A | S2-B | S2-C |

---

## 5. Definisi Metrik

| Metrik | Definisi |
|---|---|
| mAP@0.5 | Mean Average Precision pada IoU 0.5 |
| Precision, Recall | Dihitung pada confidence threshold tetap **0.5** (usulan) dan IoU 0.5. Threshold yang sama dipakai untuk ASR dan untuk ambang vanishing di loss AFOG-V. mAP tetap dihitung dengan threshold rendah (misalnya 0.05) seperti standar |
| ΔmAP | `mAP_bersih − mAP_adversarial`; juga dilaporkan sebagai % degradasi `ΔmAP / mAP_bersih × 100` |
| ASR | Proporsi objek yang **terdeteksi benar pada citra bersih** (IoU ≥ 0.5, kelas benar, conf ≥ 0.5) tetapi **gagal terdeteksi** pada citra adversarial. Definisi ini tetap dan sama untuk semua skenario. |
| FPS | Frame per detik inferensi, diukur pada GPU yang sama |
| Confidence score | Rata-rata skor deteksi yang bertahan setelah serangan |

---

## 6. Arsitektur Notebook (Kaggle `.ipynb`)

Tanpa file `.py`. Sel **SETUP** (CFG, dataset COCO, evaluasi, fungsi serangan) diletakkan identik di sel pertama setiap notebook, dilengkapi `SETUP_VERSION` dan hash `CFG`.

| No | Notebook | Akun | Isi | Output |
|---|---|---|---|---|
| 00 | `00_dataset_eksplorasi.ipynb` | Satu akun | Validasi anotasi COCO, resolusi path gambar di `images/awake` dan `images/drowsy`, konsistensi folder vs kategori anotasi, jumlah gambar/objek per kelas per split, cek kebocoran antar split, visualisasi sampel, uji sel SETUP | Laporan dataset (opsional), verifikasi jumlah gambar/objek terhadap angka di Bagian 2 |
| 01a | `01a-training-fcos.ipynb` | A | Fine-tuning FCOS sesuai Bagian 3 | `fcos_best.pth`, kurva loss/mAP |
| 01b | `01b-training-retinanet.ipynb` | B | Fine-tuning RetinaNet (`retinanet_resnet50_fpn_v2`) sesuai Bagian 3 | `retinanet_best.pth`, kurva loss/mAP |
| 02 | `02_implementasi_afog_afogv.ipynb` | Satu akun | Implementasi dan uji AFOG dan AFOG-V pada beberapa sampel | Fungsi serangan teruji (disalin ke SETUP 03) |
| 03a | `03a_serang_evaluasi_fcos.ipynb` | A | S1-A, S1-B, S1-C pada **seluruh test set (1.750 gambar)**, serangan on-the-fly tanpa menyimpan dataset adversarial | `results_fcos.csv/json`, 20-50 sampel before/after |
| 03b | `03b_serang_evaluasi_retinanet.ipynb` | B | S2-A, S2-B, S2-C | `results_retinanet.csv/json`, sampel before/after |
| 04 | `04_analisis_komparasi.ipynb` | Mana saja | Gabung hasil, % degradasi, ASR, grafik, kesimpulan | Tabel dan grafik final laporan |

**Alur data antar akun:** hanya file hasil kecil (CSV/JSON dan beberapa PNG) yang berpindah. Citra adversarial tidak disimpan. Bobot model tetap di akun masing-masing.

---

## 7. Persyaratan Fungsional

1. Dataset class membaca `annotation.coco.json` per split dan mengonversi bbox COCO (`x, y, w, h`) ke format Torchvision (`x1, y1, x2, y2`).
2. Dataset class meresolusi path gambar dari `images/awake/` dan `images/drowsy/` sesuai Bagian 2, dengan error jelas bila file tidak ditemukan atau nama file ganda.
3. Label kelas berasal dari `category_id` anotasi dengan pemetaan eksplisit (1 = `awake`, 2 = `drowsy`, 0 = background).
4. Training menyimpan checkpoint tiap epoch dan mendukung resume.
5. Evaluasi menyimpan hasil **per gambar** (jumlah deteksi, confidence, TP/FN) agar ASR dan mAP dapat dihitung ulang tanpa menyerang ulang.
6. Generate serangan mendukung resume (mencatat indeks gambar terakhir yang selesai).
7. Serangan diterapkan pada citra yang sudah dinormalisasi sesuai pipeline model, dengan clamp ke [0,1] dan proyeksi ke ε-ball.
8. Setiap notebook mencetak `SETUP_VERSION`, hash `CFG`, seed, dan jenis GPU.

## 8. Persyaratan Non-Fungsional

- **Keadilan komparasi:** semua hyperparameter, seed, transform, dan parameter serangan identik untuk kedua model.
- **Reprodusibilitas:** seed tetap, split tetap, versi library dicatat.
- **Kendala Kaggle:** sesi maksimal sekitar 12 jam, kuota GPU sekitar 30 jam/minggu per akun, output di `/kaggle/working`. Gunakan *Save & Run All (Commit)* untuk proses panjang.
- **FPS:** diukur pada satu sesi dan satu GPU yang sama (atau jenis GPU dicatat dan dijelaskan di laporan).

---

## 9. Kriteria Keberhasilan

1. Kedua model selesai dilatih dengan konfigurasi identik dan model terbaik dipilih dari validasi.
2. Keenam skenario (S1-A, S2-A, S1-B, S2-B, S1-C, S2-C) selesai dengan metrik lengkap.
3. Tersedia tabel perbandingan: mAP, ΔmAP (%), ASR, jumlah deteksi, confidence untuk FCOS vs RetinaNet pada AFOG dan AFOG-V.
4. Tersedia visualisasi before/after bounding box.
5. Kesimpulan menyebutkan arsitektur yang lebih robust beserta dukungan data.

---

## 10. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Waktu training 70 epoch melebihi kuota | Jadwal mundur | AMP, early stopping, checkpoint + resume, dua akun paralel |
| OOM pada batch 8 | Training gagal | Batch 4 + gradient accumulation 2 |
| Adaptasi AFOG ke Torchvision FCOS/RetinaNet sulit | Hasil serangan tidak valid | Notebook 02 khusus untuk uji pada sampel sebelum eksekusi massal |
| Sel SETUP berbeda antar notebook/akun | Perbandingan tidak adil | `SETUP_VERSION`, hash `CFG`, bekukan setup sebelum training |
| Kebocoran data (wajah sama di train dan test) | mAP terlalu optimistis | Cek duplikat/hash antar split di notebook 00 |
| `file_name` di JSON tidak cocok dengan path nyata (gambar di subfolder `awake/`/`drowsy/`) | Gambar gagal dimuat atau terlewat | Indeks nama file sekali di awal, verifikasi semua gambar ada di notebook 00 |
| Subfolder tidak sesuai kategori anotasi (label salah) | Training dan evaluasi bias | Cek konsistensi folder vs `category_id` di notebook 00 |
| FPS tidak sebanding antar akun | Perbandingan kecepatan bias | Ukur pada GPU sama atau catat jenis GPU |
| Sesi Kaggle putus saat serangan | Kehilangan progres | Resume berdasarkan indeks gambar + simpan CSV per gambar |

---

## 11. Urutan Pengerjaan

1. 00: validasi dataset, finalisasi sel SETUP
2. 01a dan 01b: training paralel di dua akun
3. 02: implementasi dan uji AFOG/AFOG-V (dapat dikerjakan saat training berjalan)
4. 03a dan 03b: serangan dan evaluasi paralel di dua akun
5. 04: analisis dan penarikan kesimpulan

## 12. Hal yang Perlu Dikonfirmasi

- [x] Jumlah gambar validasi dan uji (1.750 dan 1.750, masing-masing seimbang 875 awake + 875 drowsy)
- [x] Jumlah iterasi dan step size serangan: 10 iterasi, step 2/255 (usulan; cocokkan dengan paper/kode AFOG)
- [x] Confidence threshold untuk Precision, Recall, dan ASR: 0.5 (sama dengan ambang vanishing AFOG-V)
- [x] Cakupan evaluasi serangan: seluruh test set (1.750 gambar). Cadangan bila waktu serangan per gambar (diukur di notebook 02) terlalu lama: subset 500 gambar (250 awake + 250 drowsy), seed tetap dan sama di kedua akun
- [x] "Mode evaluasi statis" di proposal (Bagian 3, tahap Pelatihan Model Dasar) = model dalam mode `eval()` (*evaluation mode*). Definisi di PRD: training memakai `train()`; **serangan dan evaluasi** memakai `eval()` dengan bobot beku dan `best.pth` final. Gradien hanya dihitung terhadap citra input. Disarankan merevisi kalimat di proposal agar tidak terbaca sebagai training dalam mode eval

---

## 13. Kebutuhan Input per Notebook (Kaggle)

Dataset sudah ter-split dan berformat COCO dengan satu `annotation.coco.json` per split dan gambar di `images/awake/` serta `images/drowsy/` (lihat Bagian 2), sehingga **tidak ada dataset split terpisah** dan notebook 00 tidak lagi membuat file split.

### 13.1 Daftar Dataset/Input

Nama di bawah hanya usulan (slug sebenarnya ditentukan Kaggle). Cek path persis lewat panel **Input** atau `!ls /kaggle/input`.

| Nama (usulan) | Isi | Dibuat dari | Dipakai oleh | Perlu di akun B? |
|---|---|---|---|---|
| `nama-dataset` | Folder split (train / valid / test), masing-masing berisi `annotation.coco.json` dan `images/{awake,drowsy}/` | Upload manual (sekali) | 00, 01a/b, 02, 03a/b | Ya (share dari akun A) |
| `ckpt-fcos` | `fcos_best.pth`, `fcos_last.pth` | Output 01a, lalu "New Dataset" | 01a (resume), 02, 03a | Tidak (akun A saja) |
| `ckpt-retinanet` | `retinanet_best.pth`, `retinanet_last.pth` | Output 01b, lalu "New Dataset" | 01b (resume), 02, 03b | Tidak (akun B saja) |
| `results-fcos` | `results_fcos.csv/json`, sampel PNG | Output 03a, lalu "New Dataset" | 04 | Hanya jika 04 di akun B |
| `results-retinanet` | `results_retinanet.csv/json`, sampel PNG | Output 03b, lalu "New Dataset" | 04 | Hanya jika 04 di akun A |

Catatan:
- Dataset Kaggle private secara default. Untuk berbagi antar akun, tambahkan akun lain sebagai collaborator atau jadikan public, sehingga dataset besar tidak perlu diunggah ulang.
- Output notebook tidak otomatis menjadi input. Buka tab **Output** pada versi yang sudah selesai, klik **New Dataset**, lalu add sebagai input di sesi berikutnya.

### 13.2 Input per Notebook

#### 00_dataset_eksplorasi.ipynb (satu akun)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | Semua split |
| Internet / GPU | Off / None | Tidak diperlukan |

Isi notebook: load tiap `annotation.coco.json`, resolusi dan cek semua file gambar ada di `images/awake` dan `images/drowsy` (laporkan file hilang atau nama ganda), cek konsistensi subfolder vs `category_id`, hitung gambar dan objek per kelas (awake/drowsy) per split dan cocokkan dengan 14.000 / 1.750 / 1.750 (7.000 / 875 / 875 per kelas), validasi bbox (tidak kosong, tidak keluar batas), cek duplikat/hash antar split, visualisasi sampel, uji sel SETUP.
Output: tidak ada file yang wajib dipakai notebook lain. Hasil hitungan dicocokkan dengan angka di Bagian 2 (14.000 / 1.750 / 1.750). Notebook ini **tidak menghalangi** training, jadi 01a/01b boleh dimulai lebih dulu.

#### 01a-training-fcos.ipynb (akun A)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | Memakai `train/` dan `valid/` |
| `ckpt-fcos` (sesi ke-2 dst.) | `/kaggle/input/ckpt-fcos/fcos_last.pth` | Hanya untuk resume |
| Internet | **On** | Unduh bobot pratatih `fcos_resnet50_fpn` |
| GPU | T4 (sesuai Tabel 3.4 proposal) | Aktifkan AMP |

Output: `fcos_best.pth`, `fcos_last.pth`, log loss/mAP per epoch.

#### 01b-training-retinanet.ipynb (akun B)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | `train/` dan `valid/` |
| `ckpt-retinanet` (sesi ke-2 dst.) | `/kaggle/input/ckpt-retinanet/retinanet_last.pth` | Hanya untuk resume |
| Internet | **On** | Unduh bobot `retinanet_resnet50_fpn_v2` |
| GPU | T4 | Konfigurasi identik dengan 01a |

Output: `retinanet_best.pth`, `retinanet_last.pth`, log.

#### 02_implementasi_afog_afogv.ipynb (satu akun)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | Ambil 20-50 gambar dari `test/images/` (campur awake dan drowsy) |
| Checkpoint (opsional) | `/kaggle/input/ckpt-fcos/fcos_best.pth` | Hanya model milik akun tersebut |
| Internet | On (bila memakai bobot pratatih) | Untuk pengembangan awal |
| GPU | Ya | Serangan iteratif |

Catatan: dikerjakan **saat training berjalan** dengan bobot pratatih Torchvision, tanpa menunggu `best.pth`.
Output: fungsi AFOG dan AFOG-V yang teruji, disalin ke sel SETUP notebook 03a dan 03b.

#### 03a_serang_evaluasi_fcos.ipynb (akun A)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | `test/` |
| `ckpt-fcos` | `/kaggle/input/ckpt-fcos/fcos_best.pth` | Model terlatih |
| Internet / GPU | Off / Ya | Catat jenis GPU |

Output: `results_fcos.csv/json` (per gambar), sampel before/after (PNG), penanda resume indeks gambar. Mencakup S1-A, S1-B, S1-C.

#### 03b_serang_evaluasi_retinanet.ipynb (akun B)

| Input | Path contoh | Keterangan |
|---|---|---|
| `nama-dataset` | `/kaggle/input/nama-dataset/` | `test/` |
| `ckpt-retinanet` | `/kaggle/input/ckpt-retinanet/retinanet_best.pth` | Model terlatih |
| Internet / GPU | Off / Ya | Catat jenis GPU |

Output: `results_retinanet.csv/json`, sampel PNG. Mencakup S2-A, S2-B, S2-C.

#### 04_analisis_komparasi.ipynb (akun mana saja)

| Input | Path contoh | Keterangan |
|---|---|---|
| `results-fcos` | `/kaggle/input/results-fcos/results_fcos.csv` | Dari akun A |
| `results-retinanet` | `/kaggle/input/results-retinanet/results_retinanet.csv` | Dari akun B |
| Sampel PNG (opsional) | folder sampel dari 03a/03b | Visualisasi laporan |
| Internet / GPU | Off / None | Tidak perlu |

Output: tabel perbandingan, grafik, kesimpulan.

### 13.3 Pola Path di Sel SETUP

Slug dataset (`nama-dataset`) disesuaikan setelah menjalankan `!ls /kaggle/input`. Bila Kaggle menambah satu level folder ekstra, sesuaikan `ROOT`.

```python
import os, json

ROOT = "/kaggle/input/nama-dataset"          # sesuaikan slug
PATHS = {
    "train_dir": f"{ROOT}/train/images",
    "train_ann": f"{ROOT}/train/annotation.coco.json",
    "val_dir":   f"{ROOT}/valid/images",
    "val_ann":   f"{ROOT}/valid/annotation.coco.json",
    "test_dir":  f"{ROOT}/test/images",
    "test_ann":  f"{ROOT}/test/annotation.coco.json",
    "ckpt_in":   "/kaggle/input/ckpt-fcos",  # ckpt-retinanet di akun B
    "work":      "/kaggle/working",
}
CLASS_DIRS = ["awake", "drowsy"]

def build_image_index(images_dir):
    """Petakan nama file -> path lengkap dari images/awake dan images/drowsy."""
    index = {}
    for cls in CLASS_DIRS:
        for fn in os.listdir(os.path.join(images_dir, cls)):
            if fn in index:
                raise ValueError(f"Nama file ganda antar subfolder: {fn}")
            index[fn] = os.path.join(images_dir, cls, fn)
    return index

def resolve_path(index, file_name):
    """file_name di JSON bisa 'x.jpg' atau 'awake/x.jpg'."""
    return index[os.path.basename(file_name)]
```

### 13.4 Urutan Ketergantungan

```
nama-dataset --> 01a / 01b --> ckpt-fcos / ckpt-retinanet --> 03a / 03b --> results-* --> 04
nama-dataset --> 00 (paralel, tidak menghalangi training)
nama-dataset --> 02 (paralel dengan 01a/01b, bobot pratatih) --> kode serangan ke SETUP 03a/03b
```
