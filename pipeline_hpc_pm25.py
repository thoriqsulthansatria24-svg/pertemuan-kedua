"""
Studi Kasus: Pipeline Komputasi Paralel & Simulasi Dispersi PM2.5 Sumatera (Global API + HPC + Otomasi)
Berdasarkan kuliah tamu "Bridging Global Infrastructure to Address Local Challenge with HPC"
oleh Asst. Prof. Worawan Diaz Carballo (Department of Computer Science, Thammasat University).

Alur (sesuai pesan materi: mahasiswa sebagai integrator antara infrastruktur global, model HPC, dan domain lokal):
  1. Ambil data atmosfer global dari Open-Meteo CAMS API -> pemanfaatan Global Infrastructure
  2. Simulasi grid 2D Adveksi-Difusi PM2.5 + ukur FLOPS  -> komputasi vektor/HPC & benchmark FLOPS
  3. Olah indikator kesehatan & paparan lokal Sumatera   -> peran Integrator (HPC Ignite / APEC HPCI-MEM)
  4. Bangun dashboard web interaktif (docs/index.html)   -> visualisasi untuk pemangku kepentingan lokal
  5. Dijadwalkan GitHub Actions agar diperbarui otomatis -> alur kerja berkelanjutan (sustainable workflow)
Jika API tidak dapat diakses, dipakai DATA CONTOH sintetis (ditandai jelas di dashboard).
"""
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KOTA = {
    "Banda Aceh": (5.55, 95.32),
    "Medan": (3.59, 98.67),
    "Pekanbaru": (0.51, 101.45),
    "Padang": (-0.95, 100.35),
    "Jambi": (-1.61, 103.61),
    "Bengkulu": (-3.80, 102.26),
    "Palembang": (-2.98, 104.76),
    "Bandar Lampung": (-5.43, 105.26),
}
WIB = timezone(timedelta(hours=7))
DOCS = Path("docs")


def haversine(lat1, lon1, lat2, lon2):
    """Hitung jarak lingkaran besar (km) secara ter-vektorisasi."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def ambil_api():
    """Ambil data kualitas udara per jam (48 jam terakhir) dari Open-Meteo Global CAMS API."""
    lats = ",".join(str(v[0]) for v in KOTA.values())
    lons = ",".join(str(v[1]) for v in KOTA.values())
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lats}&longitude={lons}"
        "&hourly=pm2_5,pm10,carbon_monoxide,uv_index"
        "&past_days=2&forecast_days=1&timezone=GMT"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "studi-kasus-hpc-pm25"})
    with urllib.request.urlopen(req, timeout=25) as r:
        payload = json.load(r)

    items = payload if isinstance(payload, list) else [payload]
    rows = []
    nama_kota = list(KOTA.keys())
    for idx, item in enumerate(items):
        if idx >= len(nama_kota):
            break
        kota = nama_kota[idx]
        lat, lon = KOTA[kota]
        hourly = item.get("hourly", {})
        times = hourly.get("time", [])
        pm25_list = hourly.get("pm2_5", [])
        pm10_list = hourly.get("pm10", [])
        co_list = hourly.get("carbon_monoxide", [])
        for t_str, p25, p10, co in zip(times, pm25_list, pm10_list, co_list):
            if p25 is None:
                continue
            dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            rows.append({
                "waktu": dt,
                "kota": kota,
                "lat": lat,
                "lon": lon,
                "pm25": float(p25),
                "pm10": float(p10) if p10 is not None else float(p25) * 1.4,
                "co": float(co) if co is not None else 220.0,
            })
    if not rows:
        raise ValueError("API tidak mengembalikan data PM2.5")
    return pd.DataFrame(rows)


def data_contoh(jam=48):
    """Data sintetis atmosfer Sumatera (cadangan untuk pengujian offline)."""
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    basis_kota = {
        "Banda Aceh": 12.0, "Medan": 26.0, "Pekanbaru": 44.0, "Padang": 16.0,
        "Jambi": 38.0, "Bengkulu": 18.0, "Palembang": 41.0, "Bandar Lampung": 24.0,
    }
    rows = []
    for h in range(jam):
        dt = now - timedelta(hours=h)
        siklus_harian = 1.0 + 0.25 * np.sin(2 * np.pi * (dt.hour / 24.0))
        for kota, (lat, lon) in KOTA.items():
            pm25 = max(3.0, basis_kota[kota] * siklus_harian + rng.normal(0, 5.5))
            pm10 = pm25 * rng.uniform(1.25, 1.65)
            co = 180.0 + pm25 * rng.uniform(4.5, 7.5)
            rows.append({
                "waktu": dt, "kota": kota, "lat": lat, "lon": lon,
                "pm25": round(pm25, 1), "pm10": round(pm10, 1), "co": round(co, 0),
            })
    return pd.DataFrame(rows)


def simulasi_hpc_grid(df_terbaru):
    """
    Simulasi numerik Adveksi-Difusi Gaussian 2D konsentrasi PM2.5 di atas grid Sumatera.
    Membandingkan eksekusi Sekuensial (CPU loop) vs Vektorisasi Matriks (NumPy SIMD ala HPC)
    dan menghitung metrik FLOPS (Floating Point Operations Per Second) serta Speedup.
    """
    n_grid = 50       # 50 x 50 = 2.500 sel grid spasial Sumatera
    n_iter = 16       # 16 langkah waktu iterasi adveksi-difusi
    lat_g = np.linspace(-6.0, 6.0, n_grid)
    lon_g = np.linspace(94.5, 106.5, n_grid)
    LON, LAT = np.meshgrid(lon_g, lat_g)

    s_lat = df_terbaru["lat"].to_numpy(dtype=np.float64)
    s_lon = df_terbaru["lon"].to_numpy(dtype=np.float64)
    s_pm25 = df_terbaru["pm25"].to_numpy(dtype=np.float64)
    n_src = len(s_pm25)

    # 1. Benchmark Komputasi Sekuensial (1 iterasi diukur lalu diekstrapolasi ke n_iter)
    t0 = time.perf_counter()
    acc_seq = 0.0
    for i in range(n_grid):
        for j in range(n_grid):
            la, lo = float(LAT[i, j]), float(LON[i, j])
            val = 0.0
            for k in range(n_src):
                d2 = (la - float(s_lat[k])) ** 2 + (lo - float(s_lon[k])) ** 2
                val += float(s_pm25[k]) * float(np.exp(-d2 / 2.8))
            acc_seq += val
    waktu_seq_ms = (time.perf_counter() - t0) * n_iter * 1000.0

    # 2. Komputasi Vektor/Paralel SIMD (NumPy 3D Tensor Broadcasting untuk seluruh n_iter)
    t1 = time.perf_counter()
    dlat = LAT[:, :, None] - s_lat[None, None, :]
    dlon = LON[:, :, None] - s_lon[None, None, :]
    medan_grid = np.zeros((n_grid, n_grid), dtype=np.float64)
    for step in range(n_iter):
        # Simulasi pergeseran angin regional (adveksi) & peluruhan difusi Gaussian
        shift_lat = 0.03 * np.sin(step * 0.4)
        shift_lon = -0.04 * np.cos(step * 0.4)
        dist_sq = (dlat - shift_lat) ** 2 + (dlon - shift_lon) ** 2
        kernel = np.exp(-dist_sq / 3.2)
        bob = np.sum(kernel * s_pm25[None, None, :], axis=2) / (np.sum(kernel, axis=2) + 1e-6)
        medan_grid = 0.85 * medan_grid + 0.15 * bob
    waktu_vec_s = max(time.perf_counter() - t1, 1e-6)
    waktu_vec_ms = waktu_vec_s * 1000.0

    # Total operasi floating-point (~24 FLOPs per sel per sumber per iterasi)
    total_flops = n_grid * n_grid * n_src * n_iter * 24
    mflops = round((total_flops / waktu_vec_s) / 1e6, 1)
    speedup = round(waktu_seq_ms / max(waktu_vec_ms, 0.05), 1)

    # Subsample 15 x 15 = 225 titik grid untuk divisualisasikan pada peta interaktif
    idx = np.linspace(0, n_grid - 1, 15, dtype=int)
    grid_points = []
    for i in idx:
        for j in idx:
            nilai = round(float(medan_grid[i, j]), 1)
            grid_points.append({
                "lat": round(float(LAT[i, j]), 2),
                "lon": round(float(LON[i, j]), 2),
                "pm25": nilai,
            })

    benchmark = {
        "sel_grid": n_grid * n_grid,
        "iterasi": n_iter,
        "total_flops": int(total_flops),
        "waktu_seq_ms": round(waktu_seq_ms, 1),
        "waktu_vec_ms": round(waktu_vec_ms, 2),
        "speedup": speedup,
        "mflops": mflops,
    }
    return grid_points, benchmark


def olah(df):
    """Pengolahan spasial & klasifikasi risiko kesehatan lokal (WHO / ISPU)."""
    df = df.sort_values("waktu", ascending=False).reset_index(drop=True)
    bkl_lat, bkl_lon = KOTA["Bengkulu"]
    df["jarak_bkl"] = haversine(df["lat"], df["lon"], bkl_lat, bkl_lon).round(0).astype(int)
    df["kelas"] = pd.cut(
        df["pm25"],
        [-1, 15, 35, 55, 1000],
        labels=["Baik", "Sedang", "Sensitif", "Tidak Sehat"],
    ).astype(str)
    df["waktu"] = df["waktu"].dt.tz_convert(WIB).dt.strftime("%d %b %H:%M")
    df["pm25"] = df["pm25"].round(1)
    df["pm10"] = df["pm10"].round(1)
    df["co"] = df["co"].round(0).astype(int)
    return df[["waktu", "kota", "lat", "lon", "pm25", "pm10", "co", "kelas", "jarak_bkl"]]


def main():
    try:
        raw, sumber = ambil_api(), "Open-Meteo CAMS (data global nyata)"
    except Exception as e:
        print(f"[!] API gagal ({e}); memakai DATA CONTOH sintetis.")
        raw, sumber = data_contoh(), "DATA CONTOH (sintetis)"

    df = olah(raw)
    terbaru = df.groupby("kota", as_index=False).first()
    grid_points, bench = simulasi_hpc_grid(terbaru)

    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").touch()
    df.to_csv(DOCS / "pm25_olahan.csv", index=False)

    meta = {
        "sumber": sumber,
        "diperbarui": datetime.now(WIB).strftime("%d %b %Y %H:%M WIB"),
        "bench": bench,
    }
    html = (
        TEMPLATE.replace("__DATA__", df.head(160).to_json(orient="records"))
        .replace("__GRID__", json.dumps(grid_points))
        .replace("__META__", json.dumps(meta))
    )
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(
        f"Selesai: {len(df)} observasi ({sumber}) | "
        f"HPC Speedup {bench['speedup']}x ({bench['mflops']} MFLOPS) -> docs/index.html"
    )


TEMPLATE = r'''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Simulasi HPC Kualitas Udara (PM2.5) Sumatera</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<style>
:root{--ink:#111;--bg:#fffbee;--y:#ffd23f;--p:#ff9ecd;--g:#7ee8a2;--b:#7cc4ff;--r:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Space Grotesk","Helvetica Neue",Arial,sans-serif;line-height:1.45}
.wrap{max-width:1280px;margin:auto;padding:28px 22px 40px}
h1{margin:0;font-size:clamp(1.9rem,5.2vw,3.3rem);font-weight:800;letter-spacing:-.03em;line-height:1.05}
h1 mark{background:var(--y);padding:0 .15em;border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);display:inline-block;transform:rotate(-1deg)}
.sub{margin:14px 0 20px;font-weight:600;max-width:840px}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#fff;border:3px solid var(--ink);box-shadow:8px 8px 0 var(--ink);padding:12px 14px;margin-bottom:26px}
.bar b{font-size:.8rem;text-transform:uppercase}.bar button{font:inherit;font-weight:800;border:3px solid var(--ink);background:#fff;padding:7px 14px;cursor:pointer;box-shadow:3px 3px 0 var(--ink);transition:.1s;font-size:.85rem}
.bar button:hover{transform:translate(-2px,-2px)}.bar button.on{background:var(--y);transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink)}
.bar button.gridbtn.on{background:var(--b)}
.tag{margin-left:auto;border:3px solid var(--ink);padding:5px 12px;font-weight:700;font-size:.8rem;background:var(--g)}.tag.warn{background:var(--r)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:22px}.s3{grid-column:span 3}.s4{grid-column:span 4}.s8{grid-column:span 8}.s6{grid-column:span 6}.s12{grid-column:span 12}
.card{background:#fff;border:3px solid var(--ink);box-shadow:7px 7px 0 var(--ink);padding:16px 18px}.card h2{margin:0 0 10px;font-size:1rem;text-transform:uppercase}
.kpi{border:3px solid var(--ink);box-shadow:6px 6px 0 var(--ink);padding:14px 16px}.kpi span{font-size:.75rem;font-weight:700;text-transform:uppercase;display:block}.kpi b{font-size:1.95rem;display:block;margin-top:2px}.kpi small{font-weight:700;font-size:.78rem}
#map{height:520px;border:3px solid var(--ink)}.list{max-height:520px;overflow-y:auto;border:3px solid var(--ink)}
.row{display:flex;gap:10px;align-items:center;padding:8px 10px;border-bottom:2px solid var(--ink);font-size:.82rem}.row:last-child{border:0}
.m{min-width:52px;text-align:center;font-weight:800;border:2px solid var(--ink);padding:3px 4px}.row small{color:#444;display:block}
.br{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:.85rem;font-weight:600}.br i{display:block;height:20px;border:2px solid var(--ink);min-width:3px}.br span{min-width:118px}
.bkl{font-size:24px;line-height:24px;text-shadow:-1px -1px #fff,1px 1px #fff,1px -1px #fff,-1px 1px #fff;text-align:center}
.leg{display:flex;flex-wrap:wrap;gap:8px 14px;margin-bottom:12px;font-size:.8rem;font-weight:700;align-items:center}
.leg i{display:inline-block;width:16px;height:16px;border:2px solid var(--ink);border-radius:50%;vertical-align:-3px;margin-right:5px}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.how div{border:3px solid var(--ink);padding:10px 12px;font-size:.88rem;font-weight:500}
.how b{display:block;text-transform:uppercase;font-size:.78rem;margin-bottom:3px}.note{margin:14px 0 0;border:3px solid var(--ink);background:var(--y);padding:8px 12px;font-weight:700;font-size:.85rem}
ol{margin:0;padding-left:20px;font-weight:500;font-size:.9rem}ol li{margin-bottom:5px}code{background:var(--y);padding:0 4px;border:1px solid var(--ink)}
@media(max-width:860px){.s3{grid-column:span 6}.s4,.s8,.s6{grid-column:span 12}#map,.list{height:380px}}
</style></head><body><div class="wrap">
<h1>Simulasi <mark>HPC PM2.5</mark> Sumatera</h1>
<p class="sub">Menjembatani infrastruktur data atmosfer global &amp; komputasi matriks ter-vektorisasi (HPC) untuk memetakan tantangan polusi udara lokal di Sumatera &amp; Bengkulu. Studi kasus kuliah tamu Asst. Prof. Worawan Diaz Carballo (Thammasat University).</p>
<div class="bar"><b>Ambang PM2.5 (µg/m³)</b>
<button data-m="0" class="fbtn on">Semua (≥ 0)</button>
<button data-m="15" class="fbtn">≥ 15 (Sedang+)</button>
<button data-m="35" class="fbtn">≥ 35 (Sensitif+)</button>
<button data-m="55" class="fbtn">≥ 55 (Tidak Sehat)</button>
<button id="toggleGrid" class="gridbtn on">Grid HPC: Aktif</button>
<span class="tag" id="tag"></span></div>
<div class="grid">
<div class="kpi s3" style="background:var(--b)"><span>Rata-rata PM2.5 Sumatera</span><b id="k1"></b><small id="k1s"></small></div>
<div class="kpi s3" style="background:var(--p)"><span>Konsentrasi Tertinggi</span><b id="k2"></b><small id="k2s"></small></div>
<div class="kpi s3" style="background:var(--g)"><span>Status Kota Bengkulu</span><b id="k3"></b><small id="k3s"></small></div>
<div class="kpi s3" style="background:var(--y)"><span>Kinerja Vektor HPC (FLOPS)</span><b id="k4"></b><small id="k4s"></small></div>
<div class="card s8"><h2>Peta Stasiun &amp; Grid Simulasi Dispersi Adveksi-Difusi 2D</h2>
<div class="leg">
<span><i style="background:#7ee8a2"></i>Baik (&lt;15)</span>
<span><i style="background:#7cc4ff"></i>Sedang (15–35)</span>
<span><i style="background:#ffd23f"></i>Sensitif (35–55)</span>
<span><i style="background:#ff6b6b"></i>Tidak Sehat (≥55)</span>
<span>★ Kota Bengkulu</span>
<span>Titik kecil = Sel Grid Simulasi HPC</span>
</div>
<div id="map"></div></div>
<div class="card s4"><h2>Observasi &amp; Proyeksi Terbaru (WIB)</h2><div class="list" id="list"></div></div>
<div class="card s6"><h2>Sebaran Kategori Risiko Kesehatan (WHO)</h2><div id="b1"></div></div>
<div class="card s6"><h2>Rata-rata PM2.5 per Kota Sumatera (µg/m³)</h2><div id="b2"></div></div>
<div class="card s12"><h2>Keterkaitan dengan Materi Kuliah Tamu HPC (Worawan Diaz Carballo)</h2><div class="how">
<div style="background:var(--b)"><b>1 · Global Infrastructure → Local</b>Mengintegrasikan model kualitas udara global (Open-Meteo CAMS) untuk memantau kabut asap &amp; partikulat PM2.5 di Sumatera, serupa dengan pemodelan PM2.5 oleh superkomputer LANTA di Thailand.</div>
<div style="background:var(--g)"><b>2 · Komputasi Vektor &amp; FLOPS</b>Simulasi dispersi Gaussian 2D pada <span id="hGrid"></span> sel grid dihitung secara paralel dengan matriks ter-vektorisasi NumPy dan diukur kecepatannya dalam satuan <i>MFLOPS</i> serta <i>Speedup Factor</i>.</div>
<div style="background:var(--p)"><b>3 · Mahasiswa sebagai Integrator</b>Menerapkan konsep <i>HPC Ignite</i> &amp; kerangka <i>APEC HPCI-MEM</i>: menerjemahkan hasil simulasi numerik menjadi informasi risiko kesehatan yang mudah dipahami pakar domain lokal.</div>
<div style="background:var(--y)"><b>4 · Alur Kerja Berkelanjutan</b>Dijadwalkan berjalan otomatis setiap 6 jam melalui GitHub Actions tanpa ketergantungan server mahal, membuka akses keahlian komputasi tanpa batas geografis.</div></div>
<p class="note" id="bNote"></p></div>
<div class="card s12"><h2>Alur Komputasi &amp; Otomasi Pipeline</h2><ol>
<li><code>pipeline_hpc_pm25.py</code> mengambil data partikulat &
