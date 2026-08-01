"""
fetch_raw_data.py — Pengumpulan data mentah untuk dataset Likuiditas Saham IDX.

Skrip ini DIJALANKAN SEKALI untuk membentuk `idx_liquidity_raw/`, lalu hasilnya
di-commit ke repository. Notebook eksperimen dan `automate_Septian-Maulana.py`
membaca berkas hasil skrip ini, BUKAN memanggil internet ulang — supaya seluruh
tahapan tetap dapat direproduksi meskipun sumber daring berubah atau mati.

Dua sumber data:
  1. Profil emiten  : tabel "Daftar perusahaan yang tercatat di Bursa Efek
                      Indonesia" di Wikipedia bahasa Indonesia.
  2. Harga & volume : Yahoo Finance via pustaka `yfinance` (ticker IDX diakhiri
                      sufiks ".JK").

Keluaran:
  idx_liquidity_raw/idx_emiten_profile.csv  — 1 baris per emiten
  idx_liquidity_raw/idx_ohlcv_daily.csv     — 1 baris per emiten per hari bursa
  idx_liquidity_raw/idx_ihsg_daily.csv      — indeks IHSG (^JKSE) untuk hitung beta
  idx_liquidity_raw/_fetch_metadata.json    — jejak audit pengambilan data

Contoh pemakaian:
    python scripts/fetch_raw_data.py
    python scripts/fetch_raw_data.py --limit 30          # uji cepat
    python scripts/fetch_raw_data.py --skip-profile      # unduh harga saja
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #

# Jendela observasi dipatok eksplisit (bukan "2 tahun terakhir dari hari ini")
# supaya angka pada notebook selalu identik dengan yang dilihat reviewer.
START_DATE = "2024-08-01"
END_DATE = "2026-08-01"

WIKI_URL = (
    "https://id.wikipedia.org/wiki/"
    "Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia"
)
IHSG_TICKER = "^JKSE"

# yfinance menolak permintaan yang terlalu besar sekaligus; 40 ticker per batch
# adalah kompromi antara jumlah panggilan jaringan dan risiko rate-limit.
BATCH_SIZE = 40
BATCH_PAUSE_SEC = 1.5
MAX_RETRY = 3

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

RAW_DIR = Path(__file__).resolve().parents[1] / "idx_liquidity_raw"

# Nama bulan Indonesia -> nomor bulan, untuk mengurai "9 Desember 1997".
BULAN_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
}


# --------------------------------------------------------------------------- #
# 1. Profil emiten dari Wikipedia
# --------------------------------------------------------------------------- #

def _parse_tanggal_pencatatan(nilai: str) -> pd.Timestamp | None:
    """Ubah "9 Desember 1997 (1997-12-09)" menjadi Timestamp.

    Wikipedia menyisipkan bentuk ISO di dalam kurung. Bentuk ISO itu yang
    paling andal, jadi dipakai lebih dulu; teks Indonesia hanya jadi cadangan
    bila kurungnya tidak ada.
    """
    if not isinstance(nilai, str):
        return None

    iso = re.search(r"\((\d{4}-\d{2}-\d{2})\)", nilai)
    if iso:
        return pd.Timestamp(iso.group(1))

    teks = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", nilai)
    if teks:
        hari, nama_bulan, tahun = teks.groups()
        bulan = BULAN_ID.get(nama_bulan.lower())
        if bulan:
            return pd.Timestamp(year=int(tahun), month=bulan, day=int(hari))
    return None


def _parse_jumlah_saham(nilai: str) -> float | None:
    """Ubah "1.924.688.333" (format ribuan Indonesia) menjadi angka."""
    if not isinstance(nilai, str):
        return None
    digit = re.sub(r"[^\d]", "", nilai)
    return float(digit) if digit else None


def fetch_emiten_profile() -> pd.DataFrame:
    """Ambil dan bersihkan tabel daftar emiten IDX dari Wikipedia."""
    print(f"[1/3] Mengambil profil emiten dari Wikipedia ...")
    resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()

    tabel = pd.read_html(io.StringIO(resp.text), match="Kode", flavor="lxml")
    if not tabel:
        raise RuntimeError("Tabel emiten tidak ditemukan pada halaman Wikipedia.")
    df = tabel[0]

    df = df.rename(
        columns={
            "Kode": "kode",
            "Nama perusahaan": "nama_perusahaan",
            "Tanggal pencatatan": "tanggal_pencatatan",
            "Jumlah Saham": "jumlah_saham",
            "Papan pencatatan": "papan_pencatatan",
            "Sektor": "sektor",
        }
    )

    # Kolom "Kode" berisi "BEI: AALI" — ambil 4 huruf kodenya saja.
    df["kode"] = df["kode"].astype(str).str.extract(r"([A-Z]{4})", expand=False)

    df["tanggal_pencatatan"] = df["tanggal_pencatatan"].apply(_parse_tanggal_pencatatan)
    df["jumlah_saham"] = df["jumlah_saham"].astype(str).apply(_parse_jumlah_saham)

    for kolom in ("nama_perusahaan", "papan_pencatatan", "sektor"):
        df[kolom] = df[kolom].astype(str).str.strip()

    df = df[
        ["kode", "nama_perusahaan", "tanggal_pencatatan", "jumlah_saham",
         "papan_pencatatan", "sektor"]
    ]
    df = df.dropna(subset=["kode"]).drop_duplicates(subset=["kode"])
    df["ticker_yf"] = df["kode"] + ".JK"

    print(f"      -> {len(df)} emiten terkumpul")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Harga harian dari Yahoo Finance
# --------------------------------------------------------------------------- #

def _unduh_batch(tickers: list[str]) -> pd.DataFrame:
    """Unduh satu batch ticker dan kembalikan dalam bentuk long/tidy."""
    for percobaan in range(1, MAX_RETRY + 1):
        try:
            raw = yf.download(
                tickers,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
                timeout=60,
            )
            if raw is None or raw.empty:
                return pd.DataFrame()

            # Satu ticker menghasilkan kolom datar; banyak ticker menghasilkan
            # MultiIndex (field, ticker). Seragamkan ke MultiIndex.
            if not isinstance(raw.columns, pd.MultiIndex):
                raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

            panjang = (
                raw.stack(level=1, future_stack=True)
                .rename_axis(index=["date", "ticker_yf"])
                .reset_index()
            )
            return panjang

        except Exception as exc:  # noqa: BLE001 - sengaja luas, batch harus tahan galat
            if percobaan == MAX_RETRY:
                print(f"      !! batch gagal setelah {MAX_RETRY} percobaan: {exc}")
                return pd.DataFrame()
            tunggu = 2 ** percobaan
            print(f"      .. percobaan {percobaan} gagal ({exc}); ulang dalam {tunggu}s")
            time.sleep(tunggu)
    return pd.DataFrame()


def fetch_ohlcv(tickers: list[str]) -> pd.DataFrame:
    """Unduh OHLCV harian untuk seluruh ticker, per batch."""
    total_batch = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[2/3] Mengunduh OHLCV {len(tickers)} ticker "
          f"({START_DATE} s/d {END_DATE}) dalam {total_batch} batch ...")

    bagian: list[pd.DataFrame] = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i: i + BATCH_SIZE]
        nomor = i // BATCH_SIZE + 1
        hasil = _unduh_batch(batch)
        if not hasil.empty:
            bagian.append(hasil)
        print(f"      batch {nomor}/{total_batch}: {len(hasil):>7,} baris")
        time.sleep(BATCH_PAUSE_SEC)

    if not bagian:
        raise RuntimeError("Tidak ada data harga yang berhasil diunduh.")

    df = pd.concat(bagian, ignore_index=True)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Buang baris yang seluruh kolom harganya kosong (ticker delisting/suspend
    # tetap dikembalikan yfinance sebagai baris NaN).
    kolom_harga = [c for c in ("open", "high", "low", "close", "adj_close") if c in df]
    df = df.dropna(subset=kolom_harga, how="all")

    df["kode"] = df["ticker_yf"].str.replace(".JK", "", regex=False)
    df = df.sort_values(["kode", "date"]).reset_index(drop=True)

    print(f"      -> {len(df):,} baris, {df['kode'].nunique()} emiten punya data")
    return df


def fetch_ihsg() -> pd.DataFrame:
    """Unduh indeks IHSG sebagai acuan pasar untuk perhitungan beta."""
    print(f"[3/3] Mengunduh indeks IHSG ({IHSG_TICKER}) ...")
    raw = yf.download(
        IHSG_TICKER, start=START_DATE, end=END_DATE,
        auto_adjust=False, progress=False, timeout=60,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Data IHSG kosong.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"      -> {len(df):,} hari bursa")
    return df


# --------------------------------------------------------------------------- #
# Orkestrasi
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="batasi jumlah ticker (untuk uji cepat)")
    parser.add_argument("--skip-profile", action="store_true",
                        help="pakai profil emiten yang sudah ada di disk")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path_profile = RAW_DIR / "idx_emiten_profile.csv"

    if args.skip_profile and path_profile.exists():
        print("[1/3] Memakai profil emiten yang sudah tersimpan.")
        profile = pd.read_csv(path_profile, parse_dates=["tanggal_pencatatan"])
    else:
        profile = fetch_emiten_profile()
        profile.to_csv(path_profile, index=False)

    tickers = profile["ticker_yf"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]
        print(f"      (dibatasi {len(tickers)} ticker untuk uji cepat)")

    ohlcv = fetch_ohlcv(tickers)
    ohlcv.to_csv(RAW_DIR / "idx_ohlcv_daily.csv", index=False)

    ihsg = fetch_ihsg()
    ihsg.to_csv(RAW_DIR / "idx_ihsg_daily.csv", index=False)

    metadata = {
        "diambil_pada_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jendela_observasi": {"mulai": START_DATE, "selesai": END_DATE},
        "sumber": {"profil_emiten": WIKI_URL, "harga": "Yahoo Finance via yfinance"},
        "versi_pustaka": {"pandas": pd.__version__, "yfinance": yf.__version__},
        "jumlah_emiten_terdaftar": int(len(profile)),
        "jumlah_emiten_ada_harga": int(ohlcv["kode"].nunique()),
        "jumlah_baris_ohlcv": int(len(ohlcv)),
        "jumlah_hari_bursa_ihsg": int(len(ihsg)),
    }
    (RAW_DIR / "_fetch_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSelesai. Berkas tersimpan di", RAW_DIR)
    for berkas in sorted(RAW_DIR.iterdir()):
        print(f"  {berkas.name:<28} {berkas.stat().st_size / 1e6:>8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
