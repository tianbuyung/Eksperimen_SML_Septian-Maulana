"""
automate_Septian-Maulana.py — Otomatisasi preprocessing dataset Likuiditas Saham IDX.

Berkas ini adalah konversi dari notebook `Eksperimen_Septian-Maulana.ipynb`.
Urutan tahapannya identik dengan notebook, tetapi strukturnya berubah dari
rangkaian sel menjadi kumpulan fungsi murni yang dapat dipanggil ulang, diuji,
dan dijalankan tanpa campur tangan manual.

Urutan tahapan yang direplikasi dari notebook:

    1. Memuat tiga berkas mentah (panel harga, profil emiten, indeks IHSG)
    2. Memotong panel menjadi jendela fitur dan jendela label
    3. Merekayasa 16 fitur level emiten dari jendela fitur
    4. Membentuk label tiga kelas dari jendela label
    5. Menyaring emiten yang hari bursanya tidak memadai
    6. Membersihkan kategori sektor yang tidak valid
    7. Menangani nilai tak hingga dan nilai kosong
    8. Menghapus duplikat
    9. Memisahkan data latih dan uji secara berstrata
   10. Mengisi nilai kosong dengan median data latih
   11. Memangkas outlier pada persentil 1 dan 99 (batas dari data latih)
   12. Menerapkan transformasi log1p pada fitur yang menceng
   13. Melakukan one-hot encoding pada sektor
   14. Menstandarkan fitur numerik
   15. Memetakan label teks ke kode numerik berurut
   16. Menyimpan data siap latih beserta metadata parameter transformasi

Seluruh parameter transformasi dipelajari HANYA dari data latih, lalu diterapkan
apa adanya ke data uji.

Pemakaian sebagai skrip:

    python automate_Septian-Maulana.py
    python automate_Septian-Maulana.py --dir-raw ../idx_liquidity_raw --dir-out hasil/

Pemakaian sebagai modul:

    from automate_Septian_Maulana import jalankan_preprocessing
    hasil = jalankan_preprocessing()
    X_train, y_train = hasil["X_train"], hasil["y_train"]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Konstanta — nilainya harus persis sama dengan notebook eksperimen
# --------------------------------------------------------------------------- #

DIR_SKRIP = Path(__file__).resolve().parent
DIR_RAW_BAWAAN = DIR_SKRIP.parent / "idx_liquidity_raw"
DIR_OUT_BAWAAN = DIR_SKRIP / "idx_liquidity_preprocessing"

JENDELA_FITUR = ("2024-08-01", "2025-08-01")
JENDELA_LABEL = ("2025-08-01", "2026-08-01")

MIN_HARI_BURSA = 100

AMBANG_LIKUID = 5_000_000_000
AMBANG_MENENGAH = 500_000_000

URUTAN_KELAS = ["Tidak Likuid", "Menengah", "Likuid"]
PETA_KELAS = {nama: idx for idx, nama in enumerate(URUTAN_KELAS)}

RANDOM_STATE = 42
TEST_SIZE = 0.2

KOLOM_NUMERIK = [
    "advt_median", "advt_mean", "rasio_mean_median", "market_cap",
    "harga_median", "volume_median", "turnover_bps", "amihud_per_miliar",
    "volatilitas_tahunan", "rentang_harian_relatif", "max_drawdown", "beta_ihsg",
    "pct_hari_volume_nol", "pct_hari_harga_stagnan", "umur_listing_tahun",
    "jumlah_hari_bursa",
]
KOLOM_KATEGORIK = ["sektor"]
KOLOM_TARGET = "kelas_likuiditas"
KOLOM_IDENTITAS = ["kode", "nama_perusahaan", "papan_pencatatan", "advt_label"]

KOLOM_LOG = [
    "advt_median", "advt_mean", "market_cap", "harga_median",
    "volume_median", "turnover_bps", "amihud_per_miliar",
    "rasio_mean_median", "volatilitas_tahunan",
]

SEKTOR_VALID = {
    "Basic Materials", "Consumer Cyclicals", "Consumer Non-Cyclicals", "Energy",
    "Financials", "Healthcare", "Industrials", "Infrastructure",
    "Properties & Real Estate", "Technology", "Transportation & Logistics",
}

WINSOR_BAWAH, WINSOR_ATAS = 0.01, 0.99


# --------------------------------------------------------------------------- #
# Tahap 1-2 — memuat dan memotong data
# --------------------------------------------------------------------------- #

def muat_data_mentah(dir_raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Baca panel harga harian, profil emiten, dan indeks IHSG."""
    panel = pd.read_csv(dir_raw / "idx_ohlcv_daily.csv", parse_dates=["date"])
    profil = pd.read_csv(dir_raw / "idx_emiten_profile.csv",
                         parse_dates=["tanggal_pencatatan"])
    ihsg = pd.read_csv(dir_raw / "idx_ihsg_daily.csv", parse_dates=["date"])
    return panel, profil, ihsg


def hitung_nilai_transaksi(panel: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan nilai transaksi harian (rupiah) dan imbal hasil harian."""
    panel = panel.sort_values(["kode", "date"]).copy()
    panel["nilai_transaksi"] = panel["close"] * panel["volume"]
    panel["imbal_hasil"] = panel.groupby("kode")["close"].pct_change()
    return panel


def potong_jendela(panel: pd.DataFrame, jendela: tuple[str, str]) -> pd.DataFrame:
    """Ambil irisan panel pada rentang [mulai, selesai)."""
    mulai, selesai = jendela
    return panel[(panel["date"] >= mulai) & (panel["date"] < selesai)].copy()


# --------------------------------------------------------------------------- #
# Tahap 3 — rekayasa fitur
# --------------------------------------------------------------------------- #

def max_drawdown(harga: pd.Series) -> float:
    """Penurunan terdalam dari puncak tertinggi sebelumnya (bernilai <= 0)."""
    harga = harga.dropna()
    if len(harga) < 2:
        return np.nan
    puncak = harga.cummax()
    return float(((harga - puncak) / puncak).min())


def bangun_fitur(panel_win: pd.DataFrame, ihsg: pd.DataFrame,
                 profil: pd.DataFrame, tanggal_acuan: str) -> pd.DataFrame:
    """Ringkas panel harian satu jendela menjadi satu baris per emiten."""
    df = panel_win.copy()

    pasar = ihsg[["date", "close"]].rename(columns={"close": "ihsg_close"}).sort_values("date")
    pasar["imbal_hasil_pasar"] = pasar["ihsg_close"].pct_change()
    df = df.merge(pasar[["date", "imbal_hasil_pasar"]], on="date", how="left")

    # Amihud dinyatakan per MILIAR rupiah. Dalam satuan rupiah nilainya berorde
    # 1e-10, dan pada magnitudo sekecil itu pandas.Series.skew() keliru
    # mengembalikan 0.0 karena _zero_out_fperr menolkan jumlah kuadratnya.
    df["nilai_transaksi_pos"] = df["nilai_transaksi"].replace(0, np.nan)
    df["amihud_harian"] = df["imbal_hasil"].abs() / (df["nilai_transaksi_pos"] / 1e9)

    df["harga_stagnan"] = (df.groupby("kode")["close"].diff() == 0)
    df["rentang_relatif"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    g = df.groupby("kode")

    fitur = g.agg(
        advt_median            = ("nilai_transaksi", "median"),
        advt_mean              = ("nilai_transaksi", "mean"),
        volume_median          = ("volume", "median"),
        harga_median           = ("close", "median"),
        volatilitas_harian     = ("imbal_hasil", "std"),
        rentang_harian_relatif = ("rentang_relatif", "mean"),
        amihud_per_miliar      = ("amihud_harian", "mean"),
        jumlah_hari_bursa      = ("close", "size"),
    )

    fitur["pct_hari_volume_nol"] = g["volume"].apply(lambda s: (s.fillna(0) == 0).mean())
    fitur["pct_hari_harga_stagnan"] = g["harga_stagnan"].mean()

    def _beta(d: pd.DataFrame) -> float:
        var_pasar = d["imbal_hasil_pasar"].var()
        if not var_pasar or var_pasar <= 0:
            return np.nan
        return d["imbal_hasil"].cov(d["imbal_hasil_pasar"]) / var_pasar

    fitur["beta_ihsg"] = df.groupby("kode")[["imbal_hasil", "imbal_hasil_pasar"]].apply(_beta)
    fitur["max_drawdown"] = g["close"].apply(max_drawdown)

    fitur["volatilitas_tahunan"] = fitur["volatilitas_harian"] * np.sqrt(252)
    fitur["rasio_mean_median"] = fitur["advt_mean"] / fitur["advt_median"].replace(0, np.nan)

    fitur = fitur.drop(columns=["volatilitas_harian"]).reset_index()

    fitur = fitur.merge(
        profil[["kode", "nama_perusahaan", "jumlah_saham", "sektor",
                "papan_pencatatan", "tanggal_pencatatan"]],
        on="kode", how="left",
    )

    fitur["market_cap"] = fitur["harga_median"] * fitur["jumlah_saham"]
    # Turnover dalam basis poin agar tidak berorde 1e-4, yang membuat log1p
    # praktis tidak berefek karena log1p(x) ~ x untuk x jauh di bawah 1.
    fitur["turnover_bps"] = (
        fitur["volume_median"] / fitur["jumlah_saham"].replace(0, np.nan) * 1e4
    )

    acuan = pd.Timestamp(tanggal_acuan)
    fitur["umur_listing_tahun"] = (acuan - fitur["tanggal_pencatatan"]).dt.days / 365.25

    return fitur


# --------------------------------------------------------------------------- #
# Tahap 4-5 — pembentukan label dan penyaringan
# --------------------------------------------------------------------------- #

def ke_kelas(nilai: float) -> str:
    """Petakan median nilai transaksi harian ke kelas likuiditas."""
    if nilai >= AMBANG_LIKUID:
        return "Likuid"
    if nilai >= AMBANG_MENENGAH:
        return "Menengah"
    return "Tidak Likuid"


def bangun_label(panel_win: pd.DataFrame) -> pd.DataFrame:
    """Hitung median nilai transaksi harian pada jendela label lalu petakan ke kelas."""
    label = panel_win.groupby("kode").agg(
        advt_label       = ("nilai_transaksi", "median"),
        hari_bursa_label = ("nilai_transaksi", "size"),
    ).reset_index()
    label["kelas_likuiditas"] = label["advt_label"].apply(ke_kelas)
    return label


def gabung_dan_saring(fitur: pd.DataFrame, label: pd.DataFrame,
                      min_hari: int = MIN_HARI_BURSA) -> pd.DataFrame:
    """Gabungkan fitur dengan label, sisakan emiten yang datanya memadai."""
    data = fitur.merge(
        label[["kode", "advt_label", "hari_bursa_label", "kelas_likuiditas"]],
        on="kode", how="inner",
    )
    cukup = (data["jumlah_hari_bursa"] >= min_hari) & (data["hari_bursa_label"] >= min_hari)
    return data[cukup].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tahap 6-8 — pembersihan
# --------------------------------------------------------------------------- #

def bersihkan_sektor(data: pd.DataFrame) -> pd.DataFrame:
    """Petakan kategori sektor di luar daftar resmi IDX-IC ke 'Tidak Diketahui'."""
    data = data.copy()
    data["sektor"] = data["sektor"].where(data["sektor"].isin(SEKTOR_VALID), "Tidak Diketahui")
    data["sektor"] = data["sektor"].fillna("Tidak Diketahui")
    return data


def bersihkan_nilai_ekstrem(data: pd.DataFrame) -> pd.DataFrame:
    """Ubah nilai tak hingga menjadi kosong agar dapat diisi pada tahap berikutnya."""
    data = data.copy()
    data[KOLOM_NUMERIK] = data[KOLOM_NUMERIK].replace([np.inf, -np.inf], np.nan)
    return data


def hapus_duplikat(data: pd.DataFrame) -> pd.DataFrame:
    """Pastikan satu baris untuk satu emiten."""
    return data.drop_duplicates(subset=["kode"], keep="first").drop_duplicates()


# --------------------------------------------------------------------------- #
# Tahap 9-15 — transformasi
# --------------------------------------------------------------------------- #

def jalankan_preprocessing(
    dir_raw: Path | str = DIR_RAW_BAWAAN,
    min_hari: int = MIN_HARI_BURSA,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict[str, Any]:
    """Jalankan seluruh pipeline dari data mentah hingga data siap latih.

    Returns
    -------
    dict berisi ``X_train``, ``X_test``, ``y_train``, ``y_test``,
    ``y_train_kode``, ``y_test_kode``, ``id_train``, ``id_test``,
    dan ``metadata``.
    """
    dir_raw = Path(dir_raw)
    catat = print if verbose else (lambda *a, **k: None)

    # -- Tahap 1-2 ---------------------------------------------------------- #
    catat("[1/9] Memuat data mentah ...")
    panel, profil, ihsg = muat_data_mentah(dir_raw)
    panel = hitung_nilai_transaksi(panel)
    panel_fitur = potong_jendela(panel, JENDELA_FITUR)
    panel_label = potong_jendela(panel, JENDELA_LABEL)
    catat(f"      panel {len(panel):,} baris | "
          f"jendela fitur {panel_fitur['kode'].nunique()} emiten | "
          f"jendela label {panel_label['kode'].nunique()} emiten")

    # -- Tahap 3-5 ---------------------------------------------------------- #
    catat("[2/9] Merekayasa fitur dan membentuk label ...")
    fitur = bangun_fitur(panel_fitur, ihsg, profil, tanggal_acuan=JENDELA_FITUR[1])
    label = bangun_label(panel_label)
    data = gabung_dan_saring(fitur, label, min_hari)
    catat(f"      {len(data)} emiten lolos penyaringan hari bursa (min {min_hari} hari)")

    # -- Tahap 6-8 ---------------------------------------------------------- #
    catat("[3/9] Membersihkan sektor, nilai tak hingga, dan duplikat ...")
    data = bersihkan_sektor(data)
    data = bersihkan_nilai_ekstrem(data)
    sebelum = len(data)
    data = hapus_duplikat(data)
    catat(f"      {sebelum - len(data)} baris duplikat dihapus | "
          f"{int(data[KOLOM_NUMERIK].isna().sum().sum())} sel kosong menunggu pengisian")

    # -- Tahap 9 ------------------------------------------------------------ #
    catat("[4/9] Memisahkan data latih dan uji secara berstrata ...")
    X = data[KOLOM_NUMERIK + KOLOM_KATEGORIK].copy()
    y = data[KOLOM_TARGET].copy()
    identitas = data[KOLOM_IDENTITAS].copy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, identitas, test_size=test_size,
        random_state=random_state, stratify=y,
    )
    catat(f"      latih {len(X_train)} baris | uji {len(X_test)} baris")

    # -- Tahap 10 ----------------------------------------------------------- #
    catat("[5/9] Mengisi nilai kosong dengan median data latih ...")
    median_latih = X_train[KOLOM_NUMERIK].median()
    X_train[KOLOM_NUMERIK] = X_train[KOLOM_NUMERIK].fillna(median_latih)
    X_test[KOLOM_NUMERIK] = X_test[KOLOM_NUMERIK].fillna(median_latih)

    # -- Tahap 11 ----------------------------------------------------------- #
    catat("[6/9] Memangkas outlier pada persentil 1 dan 99 ...")
    batas_bawah = X_train[KOLOM_NUMERIK].quantile(WINSOR_BAWAH)
    batas_atas = X_train[KOLOM_NUMERIK].quantile(WINSOR_ATAS)
    dipangkas = int((
        (X_train[KOLOM_NUMERIK] < batas_bawah) | (X_train[KOLOM_NUMERIK] > batas_atas)
    ).sum().sum())
    X_train[KOLOM_NUMERIK] = X_train[KOLOM_NUMERIK].clip(batas_bawah, batas_atas, axis=1)
    X_test[KOLOM_NUMERIK] = X_test[KOLOM_NUMERIK].clip(batas_bawah, batas_atas, axis=1)
    catat(f"      {dipangkas} nilai dipangkas, jumlah baris tidak berubah")

    # -- Tahap 12 ----------------------------------------------------------- #
    catat("[7/9] Menerapkan transformasi log1p pada fitur yang menceng ...")
    for kolom in KOLOM_LOG:
        X_train[kolom] = np.log1p(X_train[kolom].clip(lower=0))
        X_test[kolom] = np.log1p(X_test[kolom].clip(lower=0))

    # -- Tahap 13 ----------------------------------------------------------- #
    catat("[8/9] One-hot encoding sektor dan standarisasi fitur numerik ...")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
    encoder.fit(X_train[KOLOM_KATEGORIK])
    nama_kolom_ohe = encoder.get_feature_names_out(KOLOM_KATEGORIK).tolist()

    def terapkan_ohe(frame: pd.DataFrame) -> pd.DataFrame:
        hasil = pd.DataFrame(
            encoder.transform(frame[KOLOM_KATEGORIK]),
            columns=nama_kolom_ohe, index=frame.index,
        )
        return pd.concat([frame.drop(columns=KOLOM_KATEGORIK), hasil], axis=1)

    X_train = terapkan_ohe(X_train)
    X_test = terapkan_ohe(X_test)

    # -- Tahap 14 ----------------------------------------------------------- #
    scaler = StandardScaler().fit(X_train[KOLOM_NUMERIK])
    X_train[KOLOM_NUMERIK] = scaler.transform(X_train[KOLOM_NUMERIK])
    X_test[KOLOM_NUMERIK] = scaler.transform(X_test[KOLOM_NUMERIK])

    # -- Tahap 15 ----------------------------------------------------------- #
    y_train_kode = y_train.map(PETA_KELAS).astype(int)
    y_test_kode = y_test.map(PETA_KELAS).astype(int)

    catat("[9/9] Selesai.")

    metadata = {
        "sumber_dataset": "IDX via yfinance + profil emiten Wikipedia",
        "jendela_fitur": {"mulai": JENDELA_FITUR[0], "selesai": JENDELA_FITUR[1]},
        "jendela_label": {"mulai": JENDELA_LABEL[0], "selesai": JENDELA_LABEL[1]},
        "min_hari_bursa": min_hari,
        "ambang_kelas_rupiah": {"likuid": AMBANG_LIKUID, "menengah": AMBANG_MENENGAH},
        "urutan_kelas": URUTAN_KELAS,
        "peta_kelas": PETA_KELAS,
        "random_state": random_state,
        "test_size": test_size,
        "kolom_numerik": KOLOM_NUMERIK,
        "kolom_log1p": KOLOM_LOG,
        "kolom_kategorik": KOLOM_KATEGORIK,
        "kolom_ohe": nama_kolom_ohe,
        "kolom_fitur_final": X_train.columns.tolist(),
        "sektor_valid": sorted(SEKTOR_VALID),
        "parameter_dipelajari_dari_data_latih": {
            "median_pengisi": median_latih.to_dict(),
            "batas_bawah_winsor_p01": batas_bawah.to_dict(),
            "batas_atas_winsor_p99": batas_atas.to_dict(),
            "scaler_mean": dict(zip(KOLOM_NUMERIK, scaler.mean_.tolist())),
            "scaler_scale": dict(zip(KOLOM_NUMERIK, scaler.scale_.tolist())),
        },
        "jumlah_baris": {"latih": int(len(X_train)), "uji": int(len(X_test))},
    }

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_kode": y_train_kode, "y_test_kode": y_test_kode,
        "id_train": id_train, "id_test": id_test,
        "encoder": encoder, "scaler": scaler,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------- #
# Tahap 16 — penyimpanan
# --------------------------------------------------------------------------- #

def simpan_hasil(hasil: dict[str, Any], dir_out: Path | str = DIR_OUT_BAWAAN,
                 verbose: bool = True) -> dict[str, Path]:
    """Tulis data latih, data uji, gabungan, dan metadata ke direktori keluaran."""
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)
    catat = print if verbose else (lambda *a, **k: None)

    def rakit(X: pd.DataFrame, y: pd.Series, y_kode: pd.Series,
              ident: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["kelas_likuiditas"] = y.values
        out["kelas_kode"] = y_kode.values
        out.insert(0, "kode", ident["kode"].values)
        return out

    train_out = rakit(hasil["X_train"], hasil["y_train"],
                      hasil["y_train_kode"], hasil["id_train"])
    test_out = rakit(hasil["X_test"], hasil["y_test"],
                     hasil["y_test_kode"], hasil["id_test"])

    jalur = {
        "train": dir_out / "idx_liquidity_train.csv",
        "test": dir_out / "idx_liquidity_test.csv",
        "gabungan": dir_out / "idx_liquidity_preprocessed.csv",
        "metadata": dir_out / "metadata_preprocessing.json",
    }

    train_out.to_csv(jalur["train"], index=False)
    test_out.to_csv(jalur["test"], index=False)
    pd.concat(
        [train_out.assign(bagian="train"), test_out.assign(bagian="test")],
        ignore_index=True,
    ).to_csv(jalur["gabungan"], index=False)
    jalur["metadata"].write_text(
        json.dumps(hasil["metadata"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    catat(f"\nBerkas tersimpan di {dir_out}")
    for nama, path in jalur.items():
        catat(f"  {path.name:<34} {path.stat().st_size / 1024:>9,.1f} KB")
    return jalur


# --------------------------------------------------------------------------- #
# Antarmuka baris perintah
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset Likuiditas Saham IDX.",
    )
    parser.add_argument("--dir-raw", type=Path, default=DIR_RAW_BAWAAN,
                        help="direktori berisi berkas mentah")
    parser.add_argument("--dir-out", type=Path, default=DIR_OUT_BAWAAN,
                        help="direktori tujuan data hasil preprocessing")
    parser.add_argument("--min-hari", type=int, default=MIN_HARI_BURSA,
                        help="minimum hari bursa per emiten di tiap jendela")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="proporsi data uji")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="benih acak untuk pemisahan data")
    parser.add_argument("--quiet", action="store_true", help="tekan keluaran log")
    args = parser.parse_args()

    if not args.dir_raw.exists():
        print(f"GALAT: direktori data mentah tidak ditemukan: {args.dir_raw}",
              file=sys.stderr)
        return 1

    verbose = not args.quiet
    hasil = jalankan_preprocessing(
        dir_raw=args.dir_raw,
        min_hari=args.min_hari,
        test_size=args.test_size,
        random_state=args.random_state,
        verbose=verbose,
    )
    simpan_hasil(hasil, dir_out=args.dir_out, verbose=verbose)

    if verbose:
        sebaran = hasil["y_train"].value_counts().reindex(URUTAN_KELAS)
        print("\nSebaran kelas pada data latih:")
        for nama, jumlah in sebaran.items():
            print(f"  {nama:<14} {jumlah:>4} ({jumlah / len(hasil['y_train']) * 100:5.1f}%)")
        print(f"\nDimensi akhir: X_train {hasil['X_train'].shape}, "
              f"X_test {hasil['X_test'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
