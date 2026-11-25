# ═══════════════════════════════════════════════════════════════
# figuras_articulo.py — Figuras del artículo EUSR-GAN (IEEE)
#
# Genera, a partir de lo que ya existe en la carpeta (pesos, logs y
# JSON por imagen), sin reentrenar nada:
#   fig2  Problema del protocolo (CAMUS nativa vs 1024×512 + tamaños)
#   fig3  Efecto del protocolo (Ahmad vs nativo, bicúbica y EUSR-GAN)
#   fig4  Resultados visuales CAMUS (+ mapas de error)
#   fig5  Resultados visuales BUSI en la lesión (+ mapas de error)
#   fig6  Mejora imagen por imagen (CAMUS nativo y BUSI)
#   fig7  Curvas de validación (EUSR vs Ahmad; BUSI: cero, CAMUS, residual)
#   fig8  Evaluación por tareas (ejemplo de detección + barras)
#   fig9  Espectro radial de potencia (BUSI)              [opcional]
#   fig10 Transferencia entre órganos (barras por zona)   [opcional]
#
# Cada figura se guarda en PNG (300 dpi) y PDF en figuras/. Si falta un
# archivo, esa figura se omite con un aviso y el resto continúa.
# Uso:  python -u figuras_articulo.py > log_figuras.txt 2>&1
# ═══════════════════════════════════════════════════════════════

import os
import re
import json
import glob
import importlib.util

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

# ── Rutas (ajustar si tus archivos tienen otro nombre) ────────
OUT = 'figuras'
LOGS = {
    'camus_eusr_ahmad': ['log_camus.txt'],                       # EUSR, protocolo Ahmad
    'camus_ahmad':      ['log_ahmad_c1.txt'],                    # Ahmad reimplementado
    'busi_b1':          ['log_busi_b1.txt', 'log_busi.txt'],     # BUSI desde cero (semilla 42)
    'busi_b2':          ['log_busi_b2.txt'],                     # BUSI desde CAMUS
    'busi_res':         ['log_busi_res.txt'],                    # BUSI salida residual
}
J_CAMUS_AHMAD  = 'resultados_camus/resultados_camus.json'
J_CAMUS_NATIVE = 'resultados_camus_native/resultados_camus.json'
J_AHMAD        = 'resultados_ahmad_camus/resultados_ahmad.json'
J_BUSI = {'B1': 'resultados_busi_native_scratch_yolo/resultados_busi.json',
          'B2': 'resultados_busi_native_camus_yolo/resultados_busi.json',
          'B0': 'resultados_busi_native_camus_yolo_zeroshot/resultados_busi.json'}
J_TASKS = 'evaluacion_tareas_busi/tareas_busi.json'

W_EUSR_CAMUS = ['resultados_camus/g_camus_stage2_best.weights.h5',
                'resultados_camus/g_camus_stage1_best.weights.h5']
W_AHMAD      = ['resultados_ahmad_camus/ahmad_gan_best.weights.h5',
                'resultados_ahmad_camus/ahmad_pre_best.weights.h5']
W_BUSI_B1    = 'resultados_busi_native_scratch_yolo/g_busi_stage1_best.weights.h5'
DETECTOR     = 'evaluacion_tareas_busi/yolo/det_hr/weights/best.pt'

OPTIONAL = True        # figuras 9 y 10

# ── Estilo sobrio, legible en escala de grises ────────────────
COL1, COL2 = 3.5, 7.16  # ancho de columna simple / doble IEEE (pulgadas)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8,
                     'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = {'hr': '#000000', 'eusr': '#1f4e9c', 'ahmad': '#c0392b', 'bic': '#7f7f7f',
     'lr': '#bdbdbd', 'b2': '#e08214', 'res': '#2ca25f'}
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print(f"  ✓ {name}")


def first(paths):
    for p in ([paths] if isinstance(paths, str) else paths):
        if os.path.exists(p):
            return p
    return None


def need(*paths):
    miss = [p for p in paths if p is None or not os.path.exists(p)]
    if miss:
        raise FileNotFoundError(', '.join(str(m) for m in miss))


def load_json(p):
    need(p)
    return json.load(open(p, encoding='utf-8'))


def load_module(path, name):
    need(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_log(paths):
    """Épocas de la Etapa 1 / preentrenamiento: (época, PSNR, SSIM). Última corrida del archivo."""
    p = first(paths)
    if p is None:
        return None
    runs, cur = [], []
    pat = re.compile(r'^\s*Época (\d+)/\d+ \|.*PSNR: ([\d.]+) dB \| SSIM: ([\d.]+)')
    for line in open(p, encoding='utf-8', errors='ignore'):
        if '[GAN]' in line or '[Perc]' in line:
            continue
        m = pat.search(line)
        if m:
            e = int(m.group(1))
            if e == 1 and cur:
                runs.append(cur)
                cur = []
            cur.append((e, float(m.group(2)), float(m.group(3))))
    if cur:
        runs.append(cur)
    return np.array(runs[-1]) if runs else None


def crop(img, cy, cx, s):
    H, W = img.shape
    y0 = int(np.clip(cy - s // 2, 0, max(0, H - s)))
    x0 = int(np.clip(cx - s // 2, 0, max(0, W - s)))
    return img[y0:y0 + s, x0:x0 + s], y0, x0


def show(ax, img, title=None, cmap='gray', vmin=0, vmax=1):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, pad=2)
    return im


def add_cbar(fig, im, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('|error|', fontsize=7)
    cb.ax.tick_params(labelsize=6)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-10))


# ═══════════════════════════════════════════════════════════════
# Módulos base (solo si existen; se cargan una vez)
# ═══════════════════════════════════════════════════════════════
_mods = {}


def mod_busi():
    if 'busi' not in _mods:
        _mods['busi'] = load_module('busi_native.py', 'busi_base')
    return _mods['busi']


def mod_camus():
    if 'camus' not in _mods:
        _mods['camus'] = load_module('ahmad_camus.py', 'camus_base')
    return _mods['camus']


def camus_test_items():
    cm = mod_camus()
    _, _, te = cm.split_patients(cm.list_camus())
    return cm, te


_busi_test = {}


def busi_test():
    if not _busi_test:
        bn = mod_busi()
        items = bn.dedupe(bn.list_busi(bn.DATASET)) if bn.DEDUP else bn.list_busi(bn.DATASET)
        _, _, te = bn.split_items(items)
        _busi_test.update(bn.load_set(te))
    return mod_busi(), _busi_test


def eusr_busi():
    if 'G_busi' not in _mods:
        need(W_BUSI_B1)
        bn = mod_busi()
        G = bn.build_generator()
        G.load_weights(W_BUSI_B1)
        _mods['G_busi'] = G
    return _mods['G_busi']


# ═══════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════
def fig2_protocol():
    """CAMUS: imagen nativa vs reescalada a 1024×512 + distribución de tamaños."""
    import nibabel as nib
    cm, te = camus_test_items()
    shapes = np.array([np.asarray(nib.load(it['img']).shape[:2])[::-1]
                       for it in cm.list_camus()])           # (filas, columnas)
    it = te[0]
    nat = cm.load_nifti(it['img'])
    big = np.asarray(Image.fromarray((nat * 255).astype(np.uint8))
                     .resize((512, 1024), Image.BICUBIC), np.float32) / 255
    fig = plt.figure(figsize=(COL2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.6, 1.6], wspace=0.6)
    a = fig.add_subplot(gs[0]); show(a, nat, f'(a) Native {nat.shape[0]}×{nat.shape[1]}')
    b = fig.add_subplot(gs[1]); show(b, big, '(b) Resized\nto 1024×512')
    c = fig.add_subplot(gs[2])
    c.scatter(shapes[:, 1], shapes[:, 0], s=4, color=C['eusr'], alpha=0.5, lw=0)
    c.add_patch(Rectangle((512, 1024), 1300, 400, color=C['bic'], alpha=0.15, lw=0))
    c.axhline(1024, color=C['ahmad'], lw=0.8, ls='--')
    c.axvline(512, color=C['ahmad'], lw=0.8, ls='--')
    c.text(530, 1040, 'larger than 1024×512\n(none)', fontsize=6.5, color=C['ahmad'], va='bottom')
    c.set_xlabel('Width (pixels)'); c.set_ylabel('Height (pixels)')
    c.set_title(f'(c) Native size of the {len(shapes)} CAMUS images')
    c.set_xlim(0, max(1250, shapes[:, 1].max() + 50)); c.set_ylim(0, 1300)
    save(fig, 'fig2_protocolo_camus')


def fig3_protocol_effect():
    a, n = load_json(J_CAMUS_AHMAD)['resumen'], load_json(J_CAMUS_NATIVE)['resumen']
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 1.5, 2.2), sharey=True)
    for ax, key, title in ((axes[0], 'psnr', '(a) Whole image'), (axes[1], 'psnr_sector', '(b) Sector only')):
        for meth, lab, col in (('eusr', 'EUSR-GAN', C['eusr']), ('bicubica', 'Bicubic', C['bic'])):
            y = [a[key][meth][0], n[key][meth][0]]
            ax.plot([0, 1], y, '-o', color=col, ms=4, lw=1.2, label=lab)
            for x, v in zip((0, 1), y):
                ax.text(x + (-0.08 if x == 0 else 0.08), v, f'{v:.2f}', ha='right' if x == 0 else 'left',
                        va='center', fontsize=6.5, color=col)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Ahmad\nprotocol', 'Native\nprotocol'])
        ax.set_xlim(-0.45, 1.45); ax.set_title(title)
    axes[0].set_ylabel('PSNR (dB), CAMUS test')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, 'fig3_efecto_protocolo')


def fig4_camus_visual():
    import tensorflow as tf  # noqa: F401
    cm, te = camus_test_items()
    rows = {r['patient']: r for r in load_json(J_CAMUS_AHMAD)['por_imagen']}
    gains = [(rows[it['patient']]['sr_psnr'] - rows[it['patient']]['bic_psnr'], it)
             for it in te if it['patient'] in rows]
    gains.sort(key=lambda t: t[0])
    it = gains[len(gains) // 2][1]                         # imagen con mejora mediana
    hr = cm.to_hr(cm.load_nifti(it['img']))
    lr = cm.degrade(hr)
    bn = mod_busi()
    Ge = bn.build_generator(); Ge.load_weights(first(W_EUSR_CAMUS))
    Ga = cm.build_ahmad_generator(); Ga.load_weights(first(W_AHMAD))
    outs = [('LR (×4 nearest)', np.kron(lr, np.ones((4, 4)))), ('Bicubic', cm.bicubic_full(lr, hr.shape)),
            ('Ahmad et al.', cm.sr_full(Ga, lr)), ('EUSR-GAN', bn.sr_tiles(Ge, lr)), ('HR', hr)]
    H, W = hr.shape
    s = min(H, W) // 3
    fig, ax = plt.subplots(2, 5, figsize=(COL2, 3.2), gridspec_kw={'hspace': 0.12, 'wspace': 0.04})
    for j, (name, img) in enumerate(outs):
        c, _, _ = crop(img, H // 2, W // 2, s)
        h, _, _ = crop(hr, H // 2, W // 2, s)
        sub = name if name == 'HR' or name.startswith('LR') else f'{name}\n{psnr(hr, img):.2f} dB'
        show(ax[0, j], c, sub)
        if name != 'HR':
            im = show(ax[1, j], np.abs(c - h), cmap='magma', vmin=0, vmax=0.2)
        else:
            ax[1, j].axis('off')
    add_cbar(fig, im, [0.915, 0.13, 0.010, 0.30])
    fig.suptitle(f'CAMUS test image ({it["patient"]}), Ahmad protocol; PSNR on the whole image', y=1.0)
    save(fig, 'fig4_visual_camus')


def fig5_busi_visual():
    bn, S = busi_test()
    G = eusr_busi()
    rows = {r['id']: r for r in load_json(J_BUSI['B1'])['por_imagen']}
    picks = []
    for cls in ('benign', 'malignant'):
        c = [(rows[it['id']]['sr_psnr_lesion'] - rows[it['id']]['bic_psnr_lesion'], i)
             for i, it in enumerate(S['items'])
             if it['clase'] == cls and it['id'] in rows and 'sr_psnr_lesion' in rows[it['id']]]
        c.sort()
        picks.append(c[len(c) // 2][1])
    fig, ax = plt.subplots(4, 4, figsize=(COL2, 7.0), gridspec_kw={'hspace': 0.15, 'wspace': 0.04})
    for r, i in enumerate(picks):
        hr, m = S['hr'][i], S['mask'][i]
        lr = bn.degrade(hr)
        outs = [('LR (×4 nearest)', np.kron(lr, np.ones((4, 4)))), ('Bicubic', bn.bicubic_full(lr, hr.shape)),
                ('EUSR-GAN', bn.sr_tiles(G, lr)), ('HR', hr)]
        ys, xs = np.nonzero(m)
        s = int(np.clip(1.6 * max(ys.ptp(), xs.ptp()), 96, min(hr.shape)))
        mc, _, _ = crop(m.astype(float), int(ys.mean()), int(xs.mean()), s)
        hc, _, _ = crop(hr, int(ys.mean()), int(xs.mean()), s)
        for j, (name, img) in enumerate(outs):
            c, _, _ = crop(img, int(ys.mean()), int(xs.mean()), s)
            show(ax[2 * r, j], c, name if r == 0 else None)
            ax[2 * r, j].contour(mc, levels=[0.5], colors='#2ca25f', linewidths=0.7)
            if name != 'HR':
                im = show(ax[2 * r + 1, j], np.abs(c - hc), cmap='magma', vmin=0, vmax=0.2)
                ax[2 * r + 1, j].set_title(f'{psnr(hc, c):.2f} dB (crop)', pad=2, fontsize=7)
            else:
                ax[2 * r + 1, j].axis('off')
        ax[2 * r, 0].set_ylabel(S['items'][i]['clase'], fontsize=8)
    add_cbar(fig, im, [0.915, 0.30, 0.010, 0.40])
    save(fig, 'fig5_visual_busi')


def fig6_per_image():
    cn = load_json(J_CAMUS_NATIVE)['por_imagen']
    b1 = load_json(J_BUSI['B1'])['por_imagen']
    groups = [('CAMUS\nwhole', [r['sr_psnr'] - r['bic_psnr'] for r in cn]),
              ('CAMUS\nsector', [r['sr_psnr_sector'] - r['bic_psnr_sector'] for r in cn]),
              ('BUSI\nwhole', [r['sr_psnr'] - r['bic_psnr'] for r in b1]),
              ('BUSI\ntissue', [r['sr_psnr_tejido'] - r['bic_psnr_tejido'] for r in b1]),
              ('BUSI\nlesion', [r['sr_psnr_lesion'] - r['bic_psnr_lesion'] for r in b1 if 'sr_psnr_lesion' in r])]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL1 * 1.3, 2.3))
    for k, (lab, v) in enumerate(groups):
        v = np.asarray(v)
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=C['eusr'], alpha=0.55, lw=0)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color='k', lw=1.2)
        ax.text(k, v.max() + 0.25, f'{np.mean(v > 0):.0%} > 0\nn = {len(v)}', ha='center', fontsize=6)
    ax.axhline(0, color=C['bic'], lw=0.8, ls='--')
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel('EUSR-GAN − bicubic PSNR (dB)')
    ax.set_ylim(min(-0.3, ax.get_ylim()[0]), ax.get_ylim()[1] + 0.6)
    save(fig, 'fig6_mejora_por_imagen')


def fig7_curves():
    e, a = parse_log(LOGS['camus_eusr_ahmad']), parse_log(LOGS['camus_ahmad'])
    pe = first(LOGS['camus_eusr_ahmad'])
    if pe and 'modo HR = native' in open(pe, encoding='utf-8', errors='ignore').read():
        print(f"    ({pe} es de la corrida NATIVA; se omite la curva de EUSR en (a))")
        e = None
    b1, b2, rs = parse_log(LOGS['busi_b1']), parse_log(LOGS['busi_b2']), parse_log(LOGS['busi_res'])
    if all(x is None for x in (e, a, b1, b2, rs)):
        raise FileNotFoundError('ningún log de entrenamiento')
    camus_ok = e is not None and a is not None
    if not camus_ok:
        print("    (sin el log de EUSR-GAN en protocolo Ahmad: la figura 7 queda solo con BUSI)")
    busi = [(d, lab, col) for d, lab, col in ((b1, 'From scratch', C['eusr']),
                                               (b2, 'CAMUS initialization', C['b2']),
                                               (rs, 'Residual output', C['res'])) if d is not None]
    n = int(camus_ok) + int(bool(busi))
    if n == 0:
        raise FileNotFoundError('logs insuficientes para la figura 7')
    fig, ax = plt.subplots(1, n, figsize=(COL2 if n == 2 else COL1, 2.3), squeeze=False)
    ax = ax[0]
    k = 0
    if camus_ok:
        for d, lab, col in ((e, 'EUSR-GAN (2.3 M)', C['eusr']), (a, 'Ahmad et al. (15.8 M)', C['ahmad'])):
            ax[k].plot(d[:, 0], d[:, 2], '-', color=col, lw=1.1, label=lab)
        ax[k].set_title('(a) CAMUS, Ahmad protocol'); ax[k].set_xlabel('Epoch')
        ax[k].set_ylabel('Validation SSIM'); ax[k].legend(frameon=False, loc='lower right')
        k += 1
    if busi:
        for d, lab, col in busi:
            ax[k].plot(d[:, 0], d[:, 2], '-', color=col, lw=1.1, label=lab)
            j = int(np.argmax(d[:, 2]))
            ax[k].plot(d[j, 0], d[j, 2], 'o', color=col, ms=3.5)
        ax[k].set_title(('(b) ' if camus_ok else '') + 'BUSI, native protocol (dot = selected epoch)')
        ax[k].set_xlabel('Epoch')
        if not camus_ok:
            ax[k].set_ylabel('Validation SSIM')
        ax[k].legend(frameon=False, loc='lower right')
        ax[k].set_ylim(max(min(d[:, 2].min() for d, _, _ in busi), 0.82), None)
    save(fig, 'fig7_curvas_validacion')


def fig8_tasks():
    T = load_json(J_TASKS)
    det = T['det']
    names = [('HR', 'HR'), ('EUSR-GAN', 'EUSR-GAN'), ('Bicúbica', 'Bicubic'), ('LR (vecino)', 'LR nearest')]
    fig = plt.figure(figsize=(COL2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.08)
    # (a)(b) ejemplo: lesión fallada por bicúbica y encontrada en EUSR-GAN
    try:
        from ultralytics import YOLO
        need(DETECTOR)
        bn, S = busi_test()
        G = eusr_busi()
        ib, ie = det['Bicúbica']['best_iou'], det['EUSR-GAN']['best_iou']
        cand = [i for i in range(len(ib)) if ib[i] is not None and ie[i] is not None
                and ib[i] < 0.5 <= ie[i]]
        i = cand[0] if cand else int(np.nanargmax([(e or 0) - (b or 0) for e, b in zip(ie, ib)]))
        hr = S['hr'][i]; lr = bn.degrade(hr)
        model = YOLO(DETECTOR)
        for k, (img, lab) in enumerate(((bn.bicubic_full(lr, hr.shape), '(a) Bicubic'),
                                        (bn.sr_tiles(G, lr), '(b) EUSR-GAN'))):
            ax = fig.add_subplot(gs[k])
            show(ax, img, lab)
            rgb = np.repeat((img * 255).astype(np.uint8)[..., None], 3, -1)
            p = model.predict(rgb, imgsz=512, conf=0.25, device='cpu', verbose=False)[0]
            for x0, y0, x1, y1 in ([] if p.boxes is None else p.boxes.xyxy.cpu().numpy()):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#ff3b30', lw=1.0))
            ax.contour(S['mask'][i].astype(float), levels=[0.5], colors='#2ca25f', linewidths=0.7)
        fig.text(0.02, -0.02, 'Green: radiologist mask · red: detector boxes', fontsize=6.5)
    except Exception as ex:
        print(f"    (ejemplo de detección omitido: {ex})")
    ax = fig.add_subplot(gs[2])
    pos = ax.get_position()  # separar (c) de la imagen (b) para que las etiquetas del eje y no la tapen
    ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    x = np.arange(len(names)); w = 0.38
    m = [det[n]['mAP50'] for n, _ in names]; r = [det[n]['recall'] for n, _ in names]
    ax.bar(x - w / 2, m, w, color=C['eusr'], label='mAP50')
    ax.bar(x + w / 2, r, w, color=C['lr'], label='Recall', edgecolor='k', lw=0.4)
    lo = max(0.0, min(m + r) - 0.1)
    for xi, (mv, rv) in enumerate(zip(m, r)):
        ax.text(xi - w / 2, mv + 0.01, f'{mv:.2f}', ha='center', fontsize=6)
        ax.text(xi + w / 2, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in names], rotation=25, ha='right', rotation_mode='anchor')
    ax.set_ylim(lo, min(1.0, max(m + r) + 0.05)); ax.set_title('(c) Lesion detection on BUSI test', pad=16)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, 'fig8_tareas')


def fig9_spectrum():
    bn, S = busi_test()
    G = eusr_busi()

    def radial(img, n=256):
        x = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), float) / 255
        x = (x - x.mean()) * np.outer(np.hanning(n), np.hanning(n))
        F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
        yy, xx = np.indices(F.shape)
        r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
        return np.log10(np.bincount(r.ravel(), F.ravel())[:n // 2] / np.maximum(np.bincount(r.ravel())[:n // 2], 1) + 1e-12)
    P = {'HR': [], 'Bicubic': [], 'EUSR-GAN': []}
    for hr in S['hr']:
        lr = bn.degrade(hr)
        P['HR'].append(radial(hr)); P['Bicubic'].append(radial(bn.bicubic_full(lr, hr.shape)))
        P['EUSR-GAN'].append(radial(bn.sr_tiles(G, lr)))
    f = np.arange(128) / 256
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for k, col in (('HR', C['hr']), ('Bicubic', C['bic']), ('EUSR-GAN', C['eusr'])):
        ax.plot(f, np.mean(P[k], 0), color=col, lw=1.1, label=k)
    ax.axvline(1 / 8, color=C['ahmad'], ls='--', lw=0.8)
    ax.text(1 / 8 + 0.005, ax.get_ylim()[1], 'LR Nyquist', fontsize=6.5, va='top', color=C['ahmad'])
    ax.set_xlabel('Spatial frequency (cycles/pixel)'); ax.set_ylabel('log$_{10}$ power')
    ax.legend(frameon=False); ax.set_title('Mean radial power spectrum, BUSI test')
    save(fig, 'fig9_espectro')


def fig10_transfer():
    zones = [('Imagen completa', 'Whole'), ('Solo tejido', 'Tissue'), ('Solo lesión', 'Lesion')]
    runs = [('B0', 'CAMUS only', C['lr']), ('B2', 'CAMUS + BUSI fine-tuning', C['b2']),
            ('B1', 'BUSI from scratch', C['eusr'])]
    data = {k: load_json(J_BUSI[k])['resumen'] for k, _, _ in runs}
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    x = np.arange(len(zones)); w = 0.26
    for j, (k, lab, col) in enumerate(runs):
        v = [data[k][z]['psnr']['delta'] for z, _ in zones]
        ax.bar(x + (j - 1) * w, v, w, color=col, label=lab, edgecolor='k', lw=0.3)
        for xi, vv in zip(x, v):
            ax.text(xi + (j - 1) * w, vv + 0.02, f'{vv:.2f}', ha='center', fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(['Whole image', 'Tissue', 'Lesion'])
    ax.set_ylabel('Gain over bicubic (dB)'); ax.set_ylim(0, 2.0)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.5, ncol=3,
               loc='upper center', bbox_to_anchor=(0.55, 1.06))
    save(fig, 'fig10_transferencia')


def main():
    figs = [fig2_protocol, fig3_protocol_effect, fig4_camus_visual, fig5_busi_visual,
            fig6_per_image, fig7_curves, fig8_tasks]
    if OPTIONAL:
        figs += [fig9_spectrum, fig10_transfer]
    for f in figs:
        print(f"{f.__name__}…", flush=True)
        try:
            f()
        except Exception as ex:
            print(f"  ✗ {f.__name__} omitida: {type(ex).__name__}: {ex}")
    print(f"\nFiguras en {os.path.abspath(OUT)}/")


if __name__ == '__main__':
    main()
