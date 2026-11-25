# ═══════════════════════════════════════════════════════════════
# busi_wavelet.py — Energía por sub-banda wavelet en BUSI nativo
#                   (análisis SIN reentrenar: solo inferencia)
#
# Pregunta: ¿la pérdida de Fourier conserva la textura fina (speckle)?
# El PSNR y el SSIM no lo miden; aquí se mide la energía de alta
# frecuencia DENTRO del tejido y de la lesión, banda por banda, con una
# transformada wavelet estacionaria (no decimada), que conserva la
# posición de cada coeficiente y permite usar las máscaras tal cual.
#
# Bandas (×4; Nyquist de la imagen LR = 0,125 ciclos/píxel):
#   nivel 1: 0,25–0,5    ciclos/píxel  → speckle más fino (no está en la LR)
#   nivel 2: 0,125–0,25  ciclos/píxel  → por encima del Nyquist de la LR
#   nivel 3: 0,0625–0,125 ciclos/píxel → contenido que la LR sí tiene
#
# Dos medidas por imagen, región y nivel, relativas a la imagen nativa:
#   energía (dB) = 10·log10(E_método / E_nativa)
#        0 dB = misma cantidad de textura; < 0 = la pierde; > 0 = la inventa
#   error   (dB) = 10·log10(E_(método − nativa) / E_nativa)
#        más negativo = textura más correcta; 0 dB = el error es tan grande
#        como la propia textura (no recupera nada útil en esa banda)
#
# Compara la bicúbica, EUSR-GAN (semillas 42, 7, 123), EUSR-GAN sin la
# pérdida de Fourier (semilla 42), EDSR-baseline (3 semillas) y
# SwinIR-light. Los modelos cuyos pesos no existan se omiten.
#
# Requiere en la misma carpeta: busi_native.py y baselines_sr.py
#   pip install PyWavelets
# Uso:   python -u busi_wavelet.py
#        WAVELET=db4 NIVELES=3 FIG_LANG=en python -u busi_wavelet.py  (valores por defecto)
# Las salidas de cada modelo se guardan en caché: si se repite el
# análisis (p. ej. con otra wavelet), no se vuelve a inferir.
# ═══════════════════════════════════════════════════════════════
import os
import json
import time

import numpy as np
from scipy import stats

try:
    import pywt
except ImportError:
    raise SystemExit("Falta PyWavelets. Instálalo con:  pip install PyWavelets")

import busi_native as bn                      # no ejecuta nada al importar

WAVELET  = os.environ.get('WAVELET', 'db4')
NIVELES  = int(os.environ.get('NIVELES', '3'))
FIG_LANG = os.environ.get('FIG_LANG', 'en')  # 'en' (artículo) | 'es'
OUT_DIR  = 'resultados_busi_wavelet'
CACHE    = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE, exist_ok=True)

# (nombre, arquitectura, carpeta de resultados, archivo de pesos)
MODELOS = [
    ('EUSR-GAN s42',    'eusr',   'resultados_busi_native_scratch_yolo',       'g_busi_stage1_best.weights.h5'),
    ('EUSR-GAN s7',     'eusr',   'resultados_busi_native_scratch_yolo_s7',    'g_busi_stage1_best.weights.h5'),
    ('EUSR-GAN s123',   'eusr',   'resultados_busi_native_scratch_yolo_s123',  'g_busi_stage1_best.weights.h5'),
    ('Sin Fourier s42', 'eusr',   'resultados_busi_native_scratch_yolo_nofft', 'g_busi_stage1_best.weights.h5'),
    ('EDSR s42',        'edsr',   'resultados_busi_native_edsr',               'edsr_busi_best.weights.h5'),
    ('EDSR s7',         'edsr',   'resultados_busi_native_edsr_s7',            'edsr_busi_best.weights.h5'),
    ('EDSR s123',       'edsr',   'resultados_busi_native_edsr_s123',          'edsr_busi_best.weights.h5'),
    ('SwinIR s42',      'swinir', 'resultados_busi_native_swinir',             'swinir_busi_best.weights.h5'),
]
SEMILLAS_EUSR = ['EUSR-GAN s42', 'EUSR-GAN s7', 'EUSR-GAN s123']
PAREJAS = [('EUSR-GAN s42', 'Sin Fourier s42'),     # efecto de la pérdida de Fourier
           ('EUSR-GAN s42', 'EDSR s42'),
           ('EUSR-GAN s42', 'Bicúbica')]
REGIONES = (('tejido', 'Tejido'), ('lesion', 'Lesión'))
EPS = 1e-12


def banda_txt(j):
    hi = 0.5 / 2 ** (j - 1)
    return f"{hi / 2:.3g}–{hi:.3g} ciclos/píxel"


# ── pesos y modelos ───────────────────────────────────────────
def ruta_pesos(carpeta, archivo):
    """Prefiere la continuación (_cont) si existe, como en el artículo."""
    for c in (f'{carpeta}_cont', carpeta):
        p = os.path.join(c, archivo)
        if os.path.exists(p):
            return p
    return None


def construir(arq):
    if arq == 'eusr':
        return bn.build_generator()
    from baselines_sr import BUILDERS
    return BUILDERS[arq](lr_size=bn.LR_PATCH)


def salidas_modelo(nombre, arq, pesos, lrs):
    """Inferencia por mosaicos (idéntica a la evaluación) con caché en disco."""
    fp = os.path.join(CACHE, nombre.replace(' ', '_') + '.npz')
    if os.path.exists(fp):
        z = np.load(fp)
        if len(z.files) == len(lrs):
            print(f"  {nombre:16s} desde caché")
            return [z[f'i{i}'].astype(np.float32) for i in range(len(lrs))]
    t0 = time.time()
    G = construir(arq)
    G.load_weights(pesos)
    out = [bn.sr_tiles(G, lr) for lr in lrs]
    np.savez_compressed(fp, **{f'i{i}': o.astype(np.float16) for i, o in enumerate(out)})
    print(f"  {nombre:16s} {pesos}  ({(time.time() - t0) / 60:.1f} min)")
    return out


# ── wavelet ───────────────────────────────────────────────────
def detalles(img):
    """SWT 2D normalizada; lista de (3, H, W) con H/V/D, nivel 1 (fino) primero."""
    c = pywt.swt2(img.astype(np.float64), WAVELET, level=NIVELES,
                  trim_approx=True, norm=True)
    return [np.stack(d) for d in c[1:][::-1]]


def medidas(det_s, det_h, m):
    r = {}
    for j, (s, h) in enumerate(zip(det_s, det_h), start=1):
        eh = float((h ** 2).sum(0)[m].mean()) + EPS
        es = float((s ** 2).sum(0)[m].mean()) + EPS
        ee = float(((s - h) ** 2).sum(0)[m].mean()) + EPS
        r[f'energia_n{j}'] = 10 * np.log10(es / eh)
        r[f'error_n{j}'] = 10 * np.log10(ee / eh)
    return r


# ── estadística ───────────────────────────────────────────────
def col(rows, met, k):
    return np.array([r[met][k] for r in rows if met in r and k in r[met]], float)


def wil(a, b):
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float('nan')


def main():
    t0 = time.time()
    items = bn.dedupe(bn.list_busi(bn.DATASET)) if bn.DEDUP else bn.list_busi(bn.DATASET)
    print("Split (idéntico al del artículo):")
    _, _, te = bn.split_items(items)
    S = bn.load_set(te)
    n = len(te)
    k = 2 ** NIVELES

    # imágenes de referencia, regiones y LR (recortadas a múltiplo de 2^niveles)
    hrs, lrs, regiones = [], [], []
    for i in range(n):
        hr = S['hr'][i]
        lrs.append(bn.degrade(hr))
        H, W = hr.shape[0] - hr.shape[0] % k, hr.shape[1] - hr.shape[1] % k
        hrs.append((H, W))
        reg = {'tejido': S['tissue'][i][:H, :W]}
        box = bn.lesion_box(S['mask'][i])
        if box is not None:
            y0, y1, x0, x1 = box
            mb = np.zeros_like(S['valid'][i])
            mb[y0:y1, x0:x1] = True
            reg['lesion'] = (mb & S['valid'][i])[:H, :W]
        regiones.append(reg)

    print(f"\nInferencia sobre {n} imágenes de prueba:")
    metodos = {'Bicúbica': [bn.bicubic_full(lr, S['hr'][i].shape) for i, lr in enumerate(lrs)]}
    for nombre, arq, carpeta, archivo in MODELOS:
        p = ruta_pesos(carpeta, archivo)
        if p is None:
            print(f"  {nombre:16s} sin pesos en {carpeta}/ → se omite")
            continue
        try:
            metodos[nombre] = salidas_modelo(nombre, arq, p, lrs)
        except Exception as e:                      # p. ej. falta baselines_sr.py
            print(f"  {nombre:16s} no se pudo evaluar ({e}) → se omite")

    print(f"\nTransformada wavelet estacionaria: '{WAVELET}', {NIVELES} niveles")
    rows = []
    for i in range(n):
        H, W = hrs[i]
        det_h = detalles(S['hr'][i][:H, :W])
        r = {'id': te[i]['id'], 'clase': te[i]['clase']}
        for nom, outs in metodos.items():
            det_s = detalles(outs[i][:H, :W])
            for reg, m in regiones[i].items():
                if m.sum() >= 64:
                    r.setdefault(reg, {})
                    r[reg].update({f'{nom}|{kk}': v for kk, v in medidas(det_s, det_h, m).items()})
        rows.append(r)

    # ── resumen ───────────────────────────────────────────────
    nombres = list(metodos)
    resumen = {}
    print("\n" + "═" * 78)
    print("ENERGÍA Y ERROR POR SUB-BANDA, RELATIVOS A LA IMAGEN NATIVA (media ± DE, dB)")
    print("  energía: 0 = igual a la nativa, < 0 = pierde textura, > 0 = la inventa")
    print("  error:   más negativo = textura más correcta")
    print("═" * 78)
    for reg, titulo in REGIONES:
        resumen[reg] = {}
        for j in range(1, NIVELES + 1):
            nn = len([r for r in rows if reg in r])
            print(f"\n{titulo} (n={nn}) · nivel {j} ({banda_txt(j)})")
            resumen[reg][j] = {}
            for nom in nombres:
                e, er = col(rows, reg, f'{nom}|energia_n{j}'), col(rows, reg, f'{nom}|error_n{j}')
                if len(e) == 0:
                    continue
                resumen[reg][j][nom] = {'energia': [float(e.mean()), float(e.std())],
                                        'error': [float(er.mean()), float(er.std())]}
                print(f"  {nom:16s} energía {e.mean():+6.2f} ± {e.std():4.2f}   "
                      f"error {er.mean():+6.2f} ± {er.std():4.2f}")

    print("\n" + "═" * 78)
    print("COMPARACIONES PAREADAS (A − B, imagen por imagen, Wilcoxon)")
    print("  |energía| menor = más cerca de la textura nativa; error menor = más correcta")
    print("═" * 78)
    pareadas = []
    for a, b in PAREJAS:
        if a not in metodos or b not in metodos:
            continue
        print(f"\n{a}  vs  {b}")
        for reg, titulo in REGIONES:
            for j in range(1, NIVELES + 1):
                sub = [r for r in rows if reg in r]
                ea, eb = (np.array([r[reg][f'{x}|energia_n{j}'] for r in sub]) for x in (a, b))
                ra, rb = (np.array([r[reg][f'{x}|error_n{j}'] for r in sub]) for x in (a, b))
                d = {'region': reg, 'nivel': j, 'A': a, 'B': b,
                     'energia_dif': float((ea - eb).mean()),
                     'A_mas_cerca_frac': float(np.mean(np.abs(ea) < np.abs(eb))),
                     'p_energia': wil(np.abs(ea), np.abs(eb)),
                     'error_dif': float((ra - rb).mean()),
                     'A_menor_error_frac': float(np.mean(ra < rb)),
                     'p_error': wil(ra, rb)}
                pareadas.append(d)
                print(f"  {titulo:6s} n{j}: energía {d['energia_dif']:+.3f} dB "
                      f"(A más cerca de la nativa en {d['A_mas_cerca_frac']:.0%}, p={d['p_energia']:.1e}) · "
                      f"error {d['error_dif']:+.3f} dB (A menor en {d['A_menor_error_frac']:.0%}, "
                      f"p={d['p_error']:.1e})")

    semillas = [s for s in SEMILLAS_EUSR if s in metodos]
    rango = {}
    if len(semillas) >= 2 and 'Sin Fourier s42' in metodos:
        print("\n" + "═" * 78)
        print(f"¿'Sin Fourier' queda fuera del rango de las {len(semillas)} semillas del modelo completo?")
        print("═" * 78)
        for reg, titulo in REGIONES:
            for j in range(1, NIVELES + 1):
                for met in ('energia', 'error'):
                    v = [resumen[reg][j][s][met][0] for s in semillas]
                    x = resumen[reg][j]['Sin Fourier s42'][met][0]
                    fuera = not (min(v) <= x <= max(v))
                    rango[f'{reg}|n{j}|{met}'] = {'semillas': v, 'sin_fourier': x, 'fuera': fuera}
                    print(f"  {titulo:6s} n{j} {met:7s}: semillas [{min(v):+.3f}, {max(v):+.3f}] · "
                          f"sin Fourier {x:+.3f} → {'FUERA' if fuera else 'dentro'}")

    figura(resumen, semillas)
    with open(os.path.join(OUT_DIR, 'wavelet_busi.json'), 'w') as fh:
        json.dump({'config': {'wavelet': WAVELET, 'niveles': NIVELES, 'n_test': n,
                              'metodos': nombres},
                   'resumen': resumen, 'pareadas': pareadas, 'rango_semillas': rango,
                   'por_imagen': rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {OUT_DIR}/")


def figura(resumen, semillas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    en = FIG_LANG == 'en'
    T = {'energia': 'Energy relative to native (dB)' if en else 'Energía relativa a la nativa (dB)',
         'error': 'Error relative to native energy (dB)' if en else 'Error relativo a la energía nativa (dB)',
         'tejido': 'Tissue' if en else 'Tejido', 'lesion': 'Lesion' if en else 'Lesión',
         'nivel': 'Wavelet level (cycles/pixel)' if en else 'Nivel wavelet (ciclos/píxel)',
         'Bicúbica': 'Bicubic' if en else 'Bicúbica',
         'EUSR': 'EUSR-GAN',
         'Sin Fourier s42': 'EUSR-GAN without FFT loss' if en else 'EUSR-GAN sin pérdida de Fourier',
         'EDSR': 'EDSR-baseline',
         'SwinIR s42': 'SwinIR-light'}
    grupos = [('Bicúbica', ['Bicúbica']), ('EUSR', semillas), ('Sin Fourier s42', ['Sin Fourier s42']),
              ('EDSR', ['EDSR s42', 'EDSR s7', 'EDSR s123']), ('SwinIR s42', ['SwinIR s42'])]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    niv = list(range(1, NIVELES + 1))
    for c, (reg, _) in enumerate(REGIONES):
        for f, met in enumerate(('energia', 'error')):
            ax = axs[f, c]
            disp = [(g, [m for m in ms if m in resumen[reg][1]]) for g, ms in grupos]
            disp = [(g, ms) for g, ms in disp if ms]
            w = 0.8 / len(disp)
            for gi, (g, ms) in enumerate(disp):
                v = np.array([[resumen[reg][j][m][met][0] for m in ms] for j in niv])
                x = np.array(niv) + (gi - (len(disp) - 1) / 2) * w
                err = None if v.shape[1] == 1 else [v.mean(1) - v.min(1), v.max(1) - v.mean(1)]
                lab = T[g] if len(ms) == 1 else f"{T[g]} ({len(ms)} {'seeds' if en else 'semillas'})"
                ax.bar(x, v.mean(1), w, yerr=err, capsize=2, label=lab, color=f'C{gi}')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_title(T[reg] if f == 0 else '')
            if c == 0:
                ax.set_ylabel(T[met])
            if f == 1:
                ax.set_xticks(niv)
                ax.set_xticklabels([f"{j}\n({0.5 / 2 ** j:.3g}–{0.5 / 2 ** (j - 1):.3g})" for j in niv])
                ax.set_xlabel(T['nivel'])
            ax.grid(axis='y', alpha=0.3)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=min(5, len(lab)), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'fig_wavelet_busi.{ext}'), dpi=200)
    plt.close(fig)
    print(f"Figura → {OUT_DIR}/fig_wavelet_busi.png / .pdf")


if __name__ == '__main__':
    main()
