# Eksperimen SML — Persistensi Likuiditas Saham Bursa Efek Indonesia

Repositori **Kriteria 1** submission kelas *Membangun Sistem Machine Learning* (Dicoding).

> **Pertanyaan penelitian:** dengan menggunakan perilaku perdagangan sebuah emiten
> selama satu tahun, dapatkah kita memprediksi kelas likuiditasnya pada tahun berikutnya?

---

## Ringkasan Dataset

| Aspek | Keterangan |
|---|---|
| Domain | Pasar modal Indonesia (Bursa Efek Indonesia) |
| Unit analisis | Emiten (satu baris per perusahaan tercatat) |
| Jumlah emiten | 888 (dari 941 emiten terdaftar) |
| Jumlah fitur | 16 fitur + 12 kolom hasil one-hot encoding sektor |
| Target | 3 kelas: `Tidak Likuid` (51,6%) / `Menengah` (30,0%) / `Likuid` (18,5%) |
| Sumber | Yahoo Finance (`yfinance`) + tabel emiten Wikipedia — **data primer** |

### Pemisahan jendela waktu

Fitur dan label dihitung dari periode yang **tidak beririsan**, sehingga tidak ada
informasi masa depan yang bocor ke dalam prediktor.

| Jendela | Periode | Peran |
|---|---|---|
| Fitur | 1 Agu 2024 – 31 Jul 2025 | seluruh variabel prediktor |
| Label | 1 Agu 2025 – 31 Jul 2026 | kelas likuiditas (target) |

### Definisi kelas

Berdasarkan **median nilai transaksi harian** pada jendela label:

| Kelas | Ambang |
|---|---|
| Likuid | ≥ Rp 5 miliar |
| Menengah | Rp 500 juta – Rp 5 miliar |
| Tidak Likuid | < Rp 500 juta |

Ambang bersifat absolut, bukan kuantil — label sebuah emiten hanya berubah bila
perilakunya benar-benar berubah, bukan karena posisi relatifnya bergeser.

### Tolok ukur yang harus dilampaui model

| Strategi naif | Akurasi |
|---|---|
| Tebak kelas mayoritas | 51,6% |
| **Salin kelas tahun sebelumnya** | **73,9%** |

Model pada Kriteria 2 baru dianggap berguna bila melampaui angka 73,9%, bukan
sekadar melampaui tebakan acak.

---

## Struktur Repositori

```text
Eksperimen_SML_Septian-Maulana
├── .github/workflows/
│   └── preprocessing.yml                    # workflow CI (kriteria Advance)
├── idx_liquidity_raw/                       # data mentah, hasil scripts/fetch_raw_data.py
│   ├── idx_emiten_profile.csv               # 941 emiten: kode, sektor, papan, jumlah saham
│   ├── idx_ohlcv_daily.csv                  # 427.567 baris panel harga harian
│   ├── idx_ihsg_daily.csv                   # indeks IHSG untuk perhitungan beta
│   └── _fetch_metadata.json                 # jejak audit pengambilan data
├── preprocessing/
│   ├── Eksperimen_Septian-Maulana.ipynb     # notebook eksperimen (mengikuti template MSML)
│   ├── automate_Septian-Maulana.py          # otomatisasi preprocessing (kriteria Skilled)
│   └── idx_liquidity_preprocessing/         # dataset siap latih
│       ├── idx_liquidity_train.csv
│       ├── idx_liquidity_test.csv
│       ├── idx_liquidity_preprocessed.csv
│       └── metadata_preprocessing.json
├── scripts/
│   └── fetch_raw_data.py                    # pengumpul data mentah (dijalankan sekali)
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### Persiapan environment

```bash
# Python 3.12.7 sesuai anjuran kelas MSML
pyenv install 3.12.7
pyenv local 3.12.7

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Menjalankan preprocessing otomatis

```bash
cd preprocessing
python automate_Septian-Maulana.py
```

Opsi lain:

```bash
python automate_Septian-Maulana.py --dir-out hasil/     # ubah direktori keluaran
python automate_Septian-Maulana.py --min-hari 120       # perketat syarat hari bursa
python automate_Septian-Maulana.py --quiet              # tanpa log
```

Sebagai modul:

```python
from automate_Septian_Maulana import jalankan_preprocessing

hasil = jalankan_preprocessing()
X_train, y_train = hasil["X_train"], hasil["y_train"]
```

### Memperbarui data mentah dari sumber

Hanya diperlukan bila ingin menyegarkan data. Prosesnya mengunduh 941 ticker dan
memakan waktu sekitar 10–15 menit.

```bash
python scripts/fetch_raw_data.py
python scripts/fetch_raw_data.py --limit 30      # uji cepat
```

---

## Pemenuhan Kriteria

| Tingkat | Ketentuan | Pemenuhan |
|---|---|---|
| **Basic** | Eksperimen manual, data loading, EDA, preprocessing di notebook | `preprocessing/Eksperimen_Septian-Maulana.ipynb` — mengikuti struktur Template Eksperimen MSML, seluruh sel berjalan tanpa error |
| **Skilled** | File `.py` yang melakukan preprocessing otomatis dengan tahapan sama | `preprocessing/automate_Septian-Maulana.py` — keluarannya identik bit-per-bit dengan notebook |
| **Advance** | Workflow GitHub Actions yang menghasilkan dataset terbaru saat trigger terpantik | `.github/workflows/preprocessing.yml` — memvalidasi hasil, mengunggah artifact, dan mengembalikan dataset ke repositori |

### Alur workflow GitHub Actions

Dipicu oleh `push`, `workflow_dispatch` (dengan opsi memperbarui data mentah), dan
jadwal mingguan setiap Senin.

```text
Checkout → Setup Python 3.12.7 → Install dependencies
   → [opsional] Perbarui data mentah dari sumber
   → Jalankan automate_Septian-Maulana.py
   → Validasi dataset (berkas ada, tidak kosong, tanpa nilai kosong, label sah)
   → Unggah artifact
   → Commit dataset terbaru kembali ke repositori
```

---

## Catatan Metodologis

**Menghindari kebocoran target.** Perumusan awal yang memprediksi likuiditas dari
periode yang sama akan bocor: target diturunkan dari nilai transaksi, sehingga fitur
nilai transaksi membuat model sekadar menghafal ambang batas. Pemisahan jendela waktu
mengubahnya menjadi persoalan peramalan yang sah.

**Kolom `papan_pencatatan` sengaja tidak dijadikan fitur.** Likuiditas rendah adalah
salah satu kriteria BEI untuk memindahkan emiten ke Papan Pemantauan Khusus, sehingga
memakainya sebagai prediktor sama dengan membocorkan jawaban. Kolom ini justru dipakai
sebagai **validasi independen**: 68,4% emiten Papan Pemantauan Khusus jatuh ke kelas
Tidak Likuid, sementara papan Utama didominasi kelas Likuid — kesesuaian yang diperoleh
tanpa kolom tersebut pernah masuk perhitungan.

**Satuan fitur dipilih dengan sengaja.** `amihud_per_miliar` dan `turnover_bps`
dinyatakan dalam satuan terskala, bukan rasio mentah. Dalam bentuk rasio, nilainya
berorde `1e-10` dan `pandas.Series.skew()` keliru mengembalikan `0.0` — fungsi internal
`_zero_out_fperr` menolkan jumlah kuadrat di bawah `1e-14`. Notebook membandingkan hasil
pandas dengan `scipy.stats.skew` sebagai pengaman terhadap kekeliruan serupa.

**Parameter transformasi hanya dipelajari dari data latih.** Median pengisi nilai
kosong, batas pemangkasan outlier, serta rata-rata dan simpangan baku untuk
standarisasi semuanya dihitung dari data latih, lalu diterapkan apa adanya ke data uji.

---

## Sumber Data

- Profil emiten — [Daftar perusahaan yang tercatat di Bursa Efek Indonesia](https://id.wikipedia.org/wiki/Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia) (Wikipedia bahasa Indonesia)
- Harga & volume harian — Yahoo Finance melalui pustaka [`yfinance`](https://github.com/ranaroussi/yfinance)
