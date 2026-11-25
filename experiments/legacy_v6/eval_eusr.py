# ═══════════════════════════════════════════════════════════════
# eval_eusr.py — Evaluación basada en tareas de EUSR-GAN v6 (BUSI)
#
# No reentrena la super-resolución: carga los pesos del generador
# entrenado por eusr_gan_v6_yolo.py y reproduce EXACTAMENTE el mismo
# split (misma semilla, mismos duplicados eliminados).
#
# Sustituye la valoración clínica por evidencia basada en las
# anotaciones de radiólogos que trae BUSI:
#   A. Fidelidad          PSNR/SSIM global y en la región de la lesión, LPIPS
#   B. Segmentación       U-Net entrenada SOLO con HR de train → Dice/IoU en test
#   C. Clasificación      benigno/maligno/normal → AUC, exactitud, cambios de clase
#   D. Detección          YOLO (mismo detector de v6) → mAP en test
#   E. Alucinaciones      falsos positivos en imágenes NORMALES (YOLO y U-Net)
#   F. Ultrasonido        CNR, SNR del speckle, nitidez del borde, espectro HF
#   G. Textura/radiómica  concordancia (CCC) de características dentro de la lesión
#
# Protocolo: cada modelo auxiliar se entrena con imágenes HR de train,
# se selecciona con val y se prueba en test con cuatro versiones de la
# misma imagen: HR (techo), Bicúbica (línea base), EUSR-GAN y LR.
# Estadística: Wilcoxon pareado + IC 95 % por bootstrap.
#
# Uso:  python eval_eusr.py      (en la misma carpeta que v6)
# ═══════════════════════════════════════════════════════════════

import os
import csv
import json
import time
import shutil
import importlib.util

import numpy as np
from PIL import Image
from scipy import ndimage, stats

# ── Configuración ─────────────────────────────────────────────
V6_FILE     = 'eusr_gan_v6_yolo.py'
GEN_WEIGHTS = None        # None → Etapa 2 si existe, si no Etapa 1
EVAL_DIR    = 'evaluacion_v6'
YOLO_BEST   = 'yolo_busi/lesion/weights/best.pt'

SR_MODE     = 'auto'      # 'auto' (elige con val) | 'tiles' | 'full'
TILE_STRIDE = 16          # solape de mosaicos en LR (parche LR = 32)

SEG_BASE     = 16         # filtros base de la U-Net
SEG_EPOCHS   = 60
SEG_BATCH    = 8
SEG_PATIENCE = 10
SEG_LR       = 1e-3

CLS_WEIGHTS = 'imagenet'  # backbone EfficientNetB0 congelado
CLS_SIZE    = 224

N_BOOT = 2000
RUN = dict(fr=True, lpips=True, seg=True, cls=True, det=True, us=True, tex=True)


# ═══════════════════════════════════════════════════════════════
# 0. CARGA DE v6 (arquitectura, split y utilidades idénticas)
# ═══════════════════════════════════════════════════════════════
def _load_v6(path):
    spec = importlib.util.spec_from_file_location('eusr_v6', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v6 = _load_v6(V6_FILE)
import tensorflow as tf   # noqa: E402  (v6 ya fijó semillas)

S     = v6.SRC_SIZE       # 256
SCALE = v6.SCALE          # 4
os.makedirs(EVAL_DIR, exist_ok=True)
RNG = np.random.default_rng(v6.SEED)


def slug(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return ''.join(c if c.isalnum() else '_' for c in name).strip('_').lower()


# ═══════════════════════════════════════════════════════════════
# 1. DATOS — mismo split que el entrenamiento
# ═══════════════════════════════════════════════════════════════
def load_split(items):
    X = np.stack([v6.load_gray(it['img']) for it in items])[..., None]
    M = np.stack([v6.load_mask(it['masks']) for it in items])[..., None]
    C = np.array([it['clase'] for it in items])
    return X.astype(np.float32), M.astype(np.float32), C


def degrade_all(X):
    return np.stack([v6.degrade(x[..., 0]) for x in X])[..., None]


def prepare_data():
    items = v6.list_busi(v6.DATASET)
    print(f"Imágenes BUSI: {len(items)}")
    if v6.DEDUP:
        items = v6.dedupe(items)
    print("Split (idéntico al de entrenamiento):")
    tr, va, te = v6.split_items(items)
    data = {}
    for name, its in (('train', tr), ('val', va), ('test', te)):
        X, M, C = load_split(its)
        data[name] = {'items': its, 'X': X, 'M': M, 'C': C}
    return data


# ═══════════════════════════════════════════════════════════════
# 2. SUPER-RESOLUCIÓN DE IMAGEN COMPLETA (64 → 256)
# ═══════════════════════════════════════════════════════════════
def bicubic_up(lr):
    return np.clip(tf.image.resize(lr, (lr.shape[1] * SCALE, lr.shape[2] * SCALE),
                                   method='bicubic').numpy(), 0.0, 1.0)


def _positions(n, t, stride):
    p = list(range(0, n - t + 1, stride))
    if p[-1] != n - t:
        p.append(n - t)
    return p


def sr_tiles(G, lr, stride=TILE_STRIDE):
    """
    Mosaicos LR de 32×32 (el mismo tamaño con que se entrenó), con
    solape y mezcla por ventana de Hann. Mantiene las estadísticas de
    GroupNorm/SE iguales a las del entrenamiento.
    """
    t = v6.LR_PATCH
    T = t * SCALE
    N, h, w, _ = lr.shape
    w1  = np.hanning(T + 2)[1:-1]
    win = np.outer(w1, w1)[None, ..., None].astype(np.float32)
    out = np.zeros((N, h * SCALE, w * SCALE, 1), np.float32)
    acc = np.zeros_like(out)
    for y in _positions(h, t, stride):
        for x in _positions(w, t, stride):
            p = v6.predict_sr(G, lr[:, y:y + t, x:x + t])
            Y, X = y * SCALE, x * SCALE
            out[:, Y:Y + T, X:X + T] += p * win
            acc[:, Y:Y + T, X:X + T] += win
    return np.clip(out / acc, 0.0, 1.0)


def make_eusr(weights, val_lr, val_hr):
    """Devuelve una función lr→sr. Elige mosaicos o imagen completa con VAL."""
    g32 = v6.build_generator(lr_size=v6.LR_PATCH)
    g32.load_weights(weights)
    g64 = v6.build_generator(lr_size=S // SCALE)
    g64.load_weights(weights)

    f_tiles = lambda lr: sr_tiles(g32, lr)                       # noqa: E731
    f_full  = lambda lr: np.clip(v6.predict_sr(g64, lr), 0, 1)   # noqa: E731
    mode = SR_MODE
    if mode == 'auto':
        p_t = v6.psnr_per_image(val_hr, f_tiles(val_lr)).mean()
        p_f = v6.psnr_per_image(val_hr, f_full(val_lr)).mean()
        mode = 'tiles' if p_t >= p_f else 'full'
        print(f"Modo SR elegido con val: {mode}  "
              f"(mosaicos {p_t:.2f} dB | completa {p_f:.2f} dB)")
    return f_tiles if mode == 'tiles' else f_full, mode


# ═══════════════════════════════════════════════════════════════
# 3. ESTADÍSTICA
# ═══════════════════════════════════════════════════════════════
def boot_ci(d, n=N_BOOT):
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float('nan'), float('nan'))
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    """a vs b pareados: diferencia media, IC 95 % bootstrap, p de Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    try:
        p = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float('nan')
    lo, hi = boot_ci(d)
    return {'diff': float(d.mean()) if len(d) else float('nan'),
            'ci_lo': lo, 'ci_hi': hi, 'p': p, 'n': int(len(d))}


def gap_recovered(sr, bic, hr):
    """Fracción de la brecha bicúbica→HR que recupera la SR."""
    den = hr - bic
    return float((sr - bic) / den) if abs(den) > 1e-9 else float('nan')


def msd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std())) if len(x) else (float('nan'),) * 2


# ═══════════════════════════════════════════════════════════════
# 4. A — FIDELIDAD (global, región de la lesión, LPIPS)
# ═══════════════════════════════════════════════════════════════
def lesion_box(mask, pad=8, min_size=32):
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min() - pad, ys.max() + pad + 1
    x0, x1 = xs.min() - pad, xs.max() + pad + 1

    def fix(a, b):
        if b - a < min_size:
            c = (a + b) // 2
            a, b = c - min_size // 2, c + min_size // 2
        a, b = max(0, a), min(S, b)
        if b - a < min_size:
            a, b = (0, min_size) if a == 0 else (S - min_size, S)
        return a, b
    y0, y1 = fix(y0, y1)
    x0, x1 = fix(x0, x1)
    return y0, y1, x0, x1


def roi_metrics(hr, sr, masks):
    ps, ss = [], []
    for h, s, m in zip(hr, sr, masks):
        b = lesion_box(m[..., 0])
        if b is None:
            ps.append(np.nan); ss.append(np.nan)
            continue
        y0, y1, x0, x1 = b
        hc, sc = h[None, y0:y1, x0:x1], s[None, y0:y1, x0:x1]
        ps.append(float(v6.psnr_per_image(hc, sc)[0]))
        ss.append(float(v6.ssim_per_image(hc, sc)[0]))
    return np.array(ps), np.array(ss)


_lpips_net = None


def lpips_scores(hr, sr):
    global _lpips_net
    import torch
    import lpips
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net='alex', verbose=False)
    to_t = lambda a: torch.from_numpy(                              # noqa: E731
        np.repeat(a.transpose(0, 3, 1, 2), 3, axis=1) * 2 - 1).float()
    out = []
    with torch.no_grad():
        for i in range(0, len(hr), 16):
            out.append(_lpips_net(to_t(hr[i:i + 16]),
                                  to_t(sr[i:i + 16])).flatten().numpy())
    return np.concatenate(out)


def eval_fidelity(V, hr, M, C, methods):
    les = C != 'normal'
    res = {}
    for name in methods:
        sr = V[name]
        p, s = v6.psnr_per_image(hr, sr), v6.ssim_per_image(hr, sr)
        pr, sr_ = roi_metrics(hr[les], sr[les], M[les])
        r = {'psnr': p, 'ssim': s, 'psnr_lesion': pr, 'ssim_lesion': sr_}
        if RUN['lpips']:
            try:
                r['lpips'] = lpips_scores(hr, sr)
            except Exception as e:
                print(f"  LPIPS omitido ({e.__class__.__name__}: {e}). "
                      "Instala con: pip install -c constraints.txt lpips")
                RUN['lpips'] = False
        res[name] = r
    return res


# ═══════════════════════════════════════════════════════════════
# 5. B — SEGMENTACIÓN (U-Net entrenada solo con HR de train)
# ═══════════════════════════════════════════════════════════════
def _cbr(x, f):
    L = tf.keras.layers
    for _ in range(2):
        x = L.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    return x


def build_unet(base=SEG_BASE):
    L = tf.keras.layers
    inp = L.Input((S, S, 1))
    x, skips = inp, []
    for f in (base, base * 2, base * 4, base * 8):
        x = _cbr(x, f)
        skips.append(x)
        x = L.MaxPooling2D()(x)
    x = _cbr(x, base * 16)
    for f, s in zip((base * 8, base * 4, base * 2, base), reversed(skips)):
        x = L.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = _cbr(L.Concatenate()([x, s]), f)
    out = L.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inp, out, name='UNet_seg')


def _dice_coef(y, p, eps=1.0):
    inter = tf.reduce_sum(y * p, axis=[1, 2, 3])
    return tf.reduce_mean((2 * inter + eps) /
                          (tf.reduce_sum(y, axis=[1, 2, 3]) +
                           tf.reduce_sum(p, axis=[1, 2, 3]) + eps))


def _seg_loss(y, p):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, p))
    return bce + (1.0 - _dice_coef(y, p))


def train_unet(Xtr, Mtr, Xva, Mva):
    path = os.path.join(EVAL_DIR, 'unet_hr.weights.h5')
    net = build_unet()
    if os.path.exists(path):
        net.load_weights(path)
        print(f"  U-Net reutilizada: {path}")
        return net

    def aug(x, m):
        if tf.random.uniform(()) > 0.5:
            x, m = tf.image.flip_left_right(x), tf.image.flip_left_right(m)
        return x, m

    ds = (tf.data.Dataset.from_tensor_slices((Xtr, Mtr))
          .shuffle(len(Xtr), seed=v6.SEED).map(aug)
          .batch(SEG_BATCH).prefetch(2))
    net.compile(optimizer=tf.keras.optimizers.Adam(SEG_LR),
                loss=_seg_loss, metrics=[_dice_coef])
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val__dice_coef', mode='max',
                                           patience=SEG_PATIENCE,
                                           restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor='val__dice_coef',
                                               mode='max', factor=0.5,
                                               patience=4, min_lr=1e-5)]
    net.fit(ds, validation_data=(Xva, Mva), epochs=SEG_EPOCHS,
            callbacks=cb, verbose=2)
    net.save_weights(path)
    return net


def dice_iou(gt, pr):
    g, p = gt > 0.5, pr > 0.5
    inter = np.logical_and(g, p).sum(axis=(1, 2, 3))
    gs, ps = g.sum(axis=(1, 2, 3)), p.sum(axis=(1, 2, 3))
    union = np.logical_or(g, p).sum(axis=(1, 2, 3))
    dice = np.where(gs + ps > 0, 2 * inter / np.maximum(gs + ps, 1), 1.0)
    iou  = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return dice, iou


def eval_segmentation(net, V, M, C, methods):
    les, nor = C != 'normal', C == 'normal'
    res, preds = {}, {}
    for name in methods:
        pr = net.predict(V[name], batch_size=8, verbose=0)
        preds[name] = pr
        d, j = dice_iou(M[les], pr[les])
        fp_area = (pr[nor] > 0.5).mean(axis=(1, 2, 3))          # fracción de imagen
        fp_any  = ((pr[nor] > 0.5).sum(axis=(1, 2, 3)) >= 50)   # ≥50 px "lesión"
        res[name] = {'dice': d, 'iou': j, 'fp_area_normal': fp_area,
                     'fp_any_normal': fp_any.astype(float)}
    return res, preds


# ═══════════════════════════════════════════════════════════════
# 6. C — CLASIFICACIÓN (EfficientNetB0 congelado + regresión logística)
# ═══════════════════════════════════════════════════════════════
_backbone = None


def cls_features(X):
    global _backbone
    if _backbone is None:
        _backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=CLS_WEIGHTS, pooling='avg',
            input_shape=(CLS_SIZE, CLS_SIZE, 3))
    out = []
    for i in range(0, len(X), 32):
        x = tf.image.resize(X[i:i + 32], (CLS_SIZE, CLS_SIZE), method='bilinear')
        x = tf.repeat(x * 255.0, 3, axis=-1)   # EfficientNet espera 0–255
        out.append(_backbone(x, training=False).numpy())
    return np.concatenate(out)


def train_classifier(Ftr, ytr, Fva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    best, best_auc = None, -1
    for c in (0.001, 0.01, 0.1, 1.0):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=c, max_iter=5000,
                                               class_weight='balanced'))
        clf.fit(Ftr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Fva), multi_class='ovr',
                            average='macro', labels=clf.classes_)
        if auc > best_auc:
            best, best_auc = clf, auc
    print(f"  Clasificador: C={best[-1].C}  AUC val (HR) = {best_auc:.3f}")
    return best


def eval_classification(clf, V, C, methods):
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
    classes = list(clf.classes_)
    res, preds = {}, {}
    for name in methods:
        P = clf.predict_proba(cls_features(V[name]))
        yhat = np.array(classes)[P.argmax(1)]
        preds[name] = (P, yhat)
        r = {'acc': accuracy_score(C, yhat),
             'bal_acc': balanced_accuracy_score(C, yhat),
             'auc_macro': roc_auc_score(C, P, multi_class='ovr',
                                        average='macro', labels=classes)}
        if 'malignant' in classes and 'benign' in classes:
            bm = np.isin(C, ['benign', 'malignant'])
            r['auc_mal_vs_ben'] = roc_auc_score(
                C[bm] == 'malignant', P[bm, classes.index('malignant')])
        res[name] = r
    # cambios de clase respecto a la predicción con HR
    for name in methods:
        res[name]['flip_vs_hr'] = float(np.mean(preds[name][1] != preds['HR'][1]))
    return res, preds


def auc_diff_ci(C, Pa, Pb, classes, n=N_BOOT):
    """IC 95 % bootstrap de AUC_macro(a) − AUC_macro(b), pareado por imagen."""
    from sklearn.metrics import roc_auc_score
    diffs = []
    for _ in range(n):
        i = RNG.integers(0, len(C), len(C))
        if len(set(C[i])) < len(classes):
            continue
        a = roc_auc_score(C[i], Pa[i], multi_class='ovr', average='macro', labels=classes)
        b = roc_auc_score(C[i], Pb[i], multi_class='ovr', average='macro', labels=classes)
        diffs.append(a - b)
    diffs = np.array(diffs)
    return (float(diffs.mean()), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float(np.mean(diffs <= 0)))


# ═══════════════════════════════════════════════════════════════
# 7. D/E — DETECCIÓN (YOLO de v6) y ALUCINACIONES en normales
# ═══════════════════════════════════════════════════════════════
def _u8_rgb(img):
    return np.repeat((np.clip(img[..., 0], 0, 1) * 255).astype(np.uint8)[..., None],
                     3, axis=-1)


def eval_detection(V, M, C, methods):
    from ultralytics import YOLO
    if not os.path.exists(YOLO_BEST):
        print(f"  No existe {YOLO_BEST}; se omite detección.")
        return {}
    model, res = YOLO(YOLO_BEST), {}
    nor = np.where(C == 'normal')[0]
    for name in methods:
        root = os.path.abspath(os.path.join(EVAL_DIR, 'yolo_data', slug(name)))
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(f'{root}/images/test'); os.makedirs(f'{root}/labels/test')
        for i, img in enumerate(V[name]):
            Image.fromarray(_u8_rgb(img)[..., 0]).save(f'{root}/images/test/{i:04d}.png')
            with open(f'{root}/labels/test/{i:04d}.txt', 'w') as fh:
                fh.write('\n'.join(v6.mask_to_yolo_lines(M[i, ..., 0] > 0.5)))
        yaml = f'{root}/data.yaml'
        with open(yaml, 'w') as fh:
            fh.write(f"path: {root}\ntrain: images/test\nval: images/test\n"
                     "test: images/test\nnc: 1\nnames: ['lesion']\n")
        m = model.val(data=yaml, split='test', imgsz=S, device='cpu', batch=16,
                      plots=False, verbose=False,
                      project=os.path.abspath(os.path.join(EVAL_DIR, 'yolo_runs')),
                      name=slug(name), exist_ok=True)
        # alucinaciones: detecciones en imágenes normales
        n_det = []
        for k in range(0, len(nor), 16):
            r = model.predict([_u8_rgb(V[name][j]) for j in nor[k:k + 16]],
                              imgsz=S, conf=v6.YOLO_CONF, device='cpu', verbose=False)
            n_det += [0 if x.boxes is None else len(x.boxes) for x in r]
        n_det = np.array(n_det, float)
        res[name] = {'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map),
                     'precision': float(m.box.mp), 'recall': float(m.box.mr),
                     'fp_per_normal': n_det, 'normal_with_fp': (n_det > 0).astype(float)}
    return res


# ═══════════════════════════════════════════════════════════════
# 8. F — MÉTRICAS PROPIAS DEL ULTRASONIDO
# ═══════════════════════════════════════════════════════════════
def _ring(mask, inner=3, outer=12):
    m = mask > 0.5
    return (ndimage.binary_dilation(m, iterations=outer) &
            ~ndimage.binary_dilation(m, iterations=inner))


def _boundary(mask):
    m = mask > 0.5
    return ndimage.binary_dilation(m, iterations=1) & ~ndimage.binary_erosion(m, iterations=1)


def us_scalar_metrics(img, mask):
    """CNR lesión–fondo, SNR del speckle en el anillo, gradiente en el borde."""
    x, m = img[..., 0], mask[..., 0] > 0.5
    if m.sum() < 30:
        return np.nan, np.nan, np.nan
    ring = _ring(mask[..., 0])
    if ring.sum() < 30:
        return np.nan, np.nan, np.nan
    l, b = x[m], x[ring]
    cnr = abs(l.mean() - b.mean()) / np.sqrt(l.var() + b.var() + 1e-8)
    snr = b.mean() / (b.std() + 1e-8)
    gm  = np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1))
    edge = gm[_boundary(mask[..., 0])].mean()
    return cnr, snr, edge


def radial_psd(img):
    x = img[..., 0] - img[..., 0].mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(x * np.outer(np.hanning(S), np.hanning(S))))) ** 2
    yy, xx = np.indices(F.shape)
    r = np.hypot(yy - S // 2, xx - S // 2).astype(int)
    prof = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return np.log10(prof[:S // 2] + 1e-12)


def eval_ultrasound(V, M, C, methods):
    les = np.where(C != 'normal')[0]
    hf0 = S // (2 * SCALE)          # frecuencias por encima de la banda de LR
    ref = {}
    res, psd_mean = {}, {}
    for name in methods:
        sc = np.array([us_scalar_metrics(V[name][i], M[i]) for i in les])
        psd = np.stack([radial_psd(img) for img in V[name]])
        psd_mean[name] = psd.mean(0)
        ref[name] = (sc, psd)
    sc_hr, psd_hr = ref['HR']
    for name in methods:
        sc, psd = ref[name]
        res[name] = {
            'cnr': sc[:, 0], 'speckle_snr': sc[:, 1], 'edge_grad': sc[:, 2],
            'cnr_abs_err': np.abs(sc[:, 0] - sc_hr[:, 0]),
            'snr_abs_err': np.abs(sc[:, 1] - sc_hr[:, 1]),
            'edge_ratio': sc[:, 2] / (sc_hr[:, 2] + 1e-8),
            'psd_hf_err': np.abs(psd[:, hf0:] - psd_hr[:, hf0:]).mean(axis=1),
        }
    return res, psd_mean


# ═══════════════════════════════════════════════════════════════
# 9. G — TEXTURA / RADIÓMICA (sin dependencias externas)
# ═══════════════════════════════════════════════════════════════
TEX_NAMES = ['mean', 'std', 'skew', 'kurt', 'entropy', 'p10', 'p90',
             'glcm_contrast', 'glcm_dissim', 'glcm_homog', 'glcm_energy', 'glcm_corr']


def _glcm(q, m, levels=32):
    P = np.zeros((levels, levels), np.float64)
    H, W = q.shape
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        a, b = q[y0:y1, x0:x1], q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        ok = m[y0:y1, x0:x1] & m[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        np.add.at(P, (a[ok], b[ok]), 1)
    P = P + P.T
    return P / max(P.sum(), 1)


def texture_features(img, mask, levels=32):
    x, m = img[..., 0], mask[..., 0] > 0.5
    v = x[m]
    if v.size < 30:
        return np.full(len(TEX_NAMES), np.nan)
    hist = np.histogram(v, bins=levels, range=(0, 1))[0] / v.size
    ent = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    q = np.clip((x * levels).astype(int), 0, levels - 1)
    P = _glcm(q, m, levels)
    i, j = np.indices(P.shape)
    mi, mj = (i * P).sum(), (j * P).sum()
    si = np.sqrt(((i - mi) ** 2 * P).sum())
    sj = np.sqrt(((j - mj) ** 2 * P).sum())
    corr = ((i - mi) * (j - mj) * P).sum() / (si * sj + 1e-12)
    return np.array([v.mean(), v.std(), stats.skew(v), stats.kurtosis(v), ent,
                     np.percentile(v, 10), np.percentile(v, 90),
                     ((i - j) ** 2 * P).sum(), (np.abs(i - j) * P).sum(),
                     (P / (1 + (i - j) ** 2)).sum(), np.sqrt((P ** 2).sum()), corr])


def ccc(x, y):
    """Coeficiente de concordancia de Lin."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12)


def eval_texture(V, M, C, methods):
    les = np.where(C != 'normal')[0]
    F = {n: np.stack([texture_features(V[n][i], M[i]) for i in les]) for n in methods}
    res = {}
    for n in methods:
        cc = np.array([ccc(F['HR'][:, k], F[n][:, k]) for k in range(len(TEX_NAMES))])
        res[n] = {'ccc': dict(zip(TEX_NAMES, cc.tolist())),
                  'ccc_mean': float(np.nanmean(cc)),
                  'n_ccc_gt_090': int(np.sum(cc > 0.90))}
    return res


# ═══════════════════════════════════════════════════════════════
# 10. REPORTE, TABLAS Y FIGURAS
# ═══════════════════════════════════════════════════════════════
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def table(title, rows, header):
    print(f"\n{title}")
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print('  ' + ' | '.join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    print('  ' + '-+-'.join('-' * x for x in w))
    for r in rows:
        print('  ' + ' | '.join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    with open(os.path.join(EVAL_DIR, slug(title) + '.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows([header] + rows)


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


def per_image_section(title, res, keys, methods, base, target, higher_better):
    """Tabla media±DE por método + comparación pareada target vs base."""
    rows = []
    for n in methods:
        rows.append([n] + [fmt(*msd(res[n][k])) for k in keys])
    table(title, rows, ['Método'] + keys)
    comp = {}
    for k in keys:
        if target in res and base in res:
            c = paired(res[target][k], res[base][k])
            c['higher_better'] = higher_better.get(k, True)
            if 'HR' in res:
                c['gap_recovered'] = gap_recovered(np.nanmean(res[target][k]),
                                                   np.nanmean(res[base][k]),
                                                   np.nanmean(res['HR'][k]))
            comp[k] = c
    if comp:
        rows = [[k, f"{c['diff']:+.4f}", f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]",
                 f"{c['p']:.2e}", fmt(c.get('gap_recovered', np.nan), d=2), c['n']]
                for k, c in comp.items()]
        table(f"{title} — {target} vs {base}", rows,
              ['métrica', 'Δ media', 'IC 95 %', 'p Wilcoxon', 'brecha recuperada', 'n'])
    return comp


def figure_examples(V, M, C, seg_preds, methods, n=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    les = np.where(C != 'normal')[0][:n]
    show = [m for m in methods if m != 'LR']
    fig, ax = plt.subplots(len(les), len(show), figsize=(3.2 * len(show), 3.2 * len(les)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(les):
        for c, name in enumerate(show):
            a = ax[r, c]
            a.imshow(V[name][i, ..., 0], cmap='gray', vmin=0, vmax=1)
            a.contour(M[i, ..., 0], levels=[0.5], colors='lime', linewidths=1)
            if seg_preds is not None:
                a.contour(seg_preds[name][i, ..., 0], levels=[0.5], colors='red', linewidths=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(name)
        ax[r, 0].set_ylabel(C[i])
    fig.suptitle('Verde: máscara del radiólogo · Rojo: U-Net (entrenada con HR)')
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, 'ejemplos_segmentacion.png'), dpi=150)
    plt.close(fig)


def figure_psd(psd_mean, methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(6, 4))
    f = np.arange(S // 2) / S
    for n in methods:
        a.plot(f, psd_mean[n], label=n, lw=1.4)
    a.axvline(1 / (2 * SCALE), color='gray', ls='--', lw=1)
    a.text(1 / (2 * SCALE), a.get_ylim()[1], ' límite LR', va='top', fontsize=8)
    a.set_xlabel('frecuencia (ciclos/píxel)')
    a.set_ylabel('log10 densidad espectral')
    a.legend(frameon=False)
    a.set_title('Espectro radial promedio (test)')
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, 'espectro_radial.png'), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    data = prepare_data()
    Xte, Mte, Cte = data['test']['X'], data['test']['M'], data['test']['C']
    print(f"Test: {len(Xte)} imágenes  "
          + "  ".join(f"{c}={int((Cte == c).sum())}" for c in sorted(set(Cte))))

    # ── Pesos del generador ─────────────────────────────────────
    w = GEN_WEIGHTS or (v6.W_S2 if os.path.exists(v6.W_S2) else v6.W_S1)
    print(f"Pesos del generador: {w}")

    # ── Versiones de cada imagen de test ────────────────────────
    lr_va = degrade_all(data['val']['X'])
    eusr, mode = make_eusr(w, lr_va, data['val']['X'])
    lr_te = degrade_all(Xte)
    V = {'HR': Xte,
         'Bicúbica': bicubic_up(lr_te),
         'EUSR-GAN': eusr(lr_te),
         'LR': np.clip(tf.image.resize(lr_te, (S, S), method='nearest').numpy(), 0, 1)}
    # Para agregar otro método (Ahmad et al., ESRGAN, SwinIR…):
    #   V['Ahmad et al.'] = funcion_sr(lr_te)    # (N,64,64,1) → (N,256,256,1)
    methods = list(V.keys())
    base, target = 'Bicúbica', 'EUSR-GAN'

    sr_dir = os.path.join(EVAL_DIR, 'sr_test')
    os.makedirs(sr_dir, exist_ok=True)
    for i, it in enumerate(data['test']['items']):
        Image.fromarray((V[target][i, ..., 0] * 255).astype(np.uint8)).save(
            os.path.join(sr_dir, slug(it['id']) + '.png'))

    report = {'weights': w, 'sr_mode': mode, 'n_test': int(len(Xte)),
              'classes_test': {c: int((Cte == c).sum()) for c in sorted(set(Cte))}}
    fr_methods = [m for m in methods if m != 'HR']

    # ── A. Fidelidad ────────────────────────────────────────────
    if RUN['fr']:
        print("\n[A] Fidelidad")
        fr = eval_fidelity(V, Xte, Mte, Cte, fr_methods)
        keys = ['psnr', 'ssim', 'psnr_lesion', 'ssim_lesion'] + (['lpips'] if RUN['lpips'] else [])
        comp = per_image_section('A Fidelidad', fr, keys, fr_methods, base, target,
                                 {'lpips': False})
        report['fidelidad'] = {'por_metodo': {n: {k: msd(fr[n][k]) for k in keys} for n in fr_methods},
                               'comparacion': comp}

    # ── B. Segmentación ─────────────────────────────────────────
    seg_preds = None
    if RUN['seg']:
        print("\n[B] Segmentación — entrenando U-Net con HR de train")
        net = train_unet(data['train']['X'], data['train']['M'],
                         data['val']['X'], data['val']['M'])
        seg, seg_preds = eval_segmentation(net, V, Mte, Cte, methods)
        comp = per_image_section('B Segmentacion', seg,
                                 ['dice', 'iou', 'fp_area_normal', 'fp_any_normal'],
                                 methods, base, target,
                                 {'fp_area_normal': False, 'fp_any_normal': False})
        report['segmentacion'] = {'por_metodo': {n: {k: msd(v) for k, v in seg[n].items()} for n in methods},
                                  'comparacion': comp}

    # ── C. Clasificación ────────────────────────────────────────
    if RUN['cls']:
        print("\n[C] Clasificación — características EfficientNetB0 (congelado)")
        clf = train_classifier(cls_features(data['train']['X']), data['train']['C'],
                               cls_features(data['val']['X']), data['val']['C'])
        cls, cpred = eval_classification(clf, V, Cte, methods)
        rows = [[n] + [fmt(cls[n][k]) for k in cls[n]] for n in methods]
        table('C Clasificacion', rows, ['Método'] + list(cls[methods[0]].keys()))
        d, lo, hi, p = auc_diff_ci(Cte, cpred[target][0], cpred[base][0], list(clf.classes_))
        print(f"  ΔAUC macro {target} − {base}: {d:+.3f}  IC 95 % [{lo:+.3f}, {hi:+.3f}]  "
              f"p(bootstrap, una cola) = {p:.3f}")
        report['clasificacion'] = {'por_metodo': cls,
                                   'delta_auc': {'diff': d, 'ci_lo': lo, 'ci_hi': hi, 'p': p}}

    # ── D/E. Detección y alucinaciones ──────────────────────────
    if RUN['det']:
        print("\n[D] Detección (YOLO de v6) y [E] alucinaciones en normales")
        det = eval_detection(V, Mte, Cte, methods)
        if det:
            rows = [[n, fmt(det[n]['mAP50']), fmt(det[n]['mAP50_95']),
                     fmt(det[n]['precision']), fmt(det[n]['recall']),
                     fmt(*msd(det[n]['fp_per_normal']), d=2),
                     f"{np.mean(det[n]['normal_with_fp']):.0%}"] for n in methods]
            table('D Deteccion', rows, ['Método', 'mAP50', 'mAP50-95', 'P', 'R',
                                        'FP/normal', 'normales con FP'])
            report['deteccion'] = {n: {k: (v if not isinstance(v, np.ndarray) else msd(v))
                                       for k, v in det[n].items()} for n in methods}

    # ── F. Ultrasonido ──────────────────────────────────────────
    if RUN['us']:
        print("\n[F] Métricas de ultrasonido")
        us, psd_mean = eval_ultrasound(V, Mte, Cte, methods)
        keys = ['cnr', 'speckle_snr', 'edge_grad', 'cnr_abs_err', 'snr_abs_err',
                'edge_ratio', 'psd_hf_err']
        comp = per_image_section('F Ultrasonido', us, keys, methods, base, target,
                                 {'cnr_abs_err': False, 'snr_abs_err': False,
                                  'psd_hf_err': False})
        figure_psd(psd_mean, methods)
        report['ultrasonido'] = {'por_metodo': {n: {k: msd(us[n][k]) for k in keys} for n in methods},
                                 'comparacion': comp}

    # ── G. Textura ──────────────────────────────────────────────
    if RUN['tex']:
        print("\n[G] Textura / radiómica (concordancia con HR)")
        tex = eval_texture(V, Mte, Cte, fr_methods + ['HR'])
        rows = [[n, fmt(tex[n]['ccc_mean']), f"{tex[n]['n_ccc_gt_090']}/{len(TEX_NAMES)}"]
                for n in fr_methods]
        table('G Textura', rows, ['Método', 'CCC medio', 'características con CCC>0.90'])
        report['textura'] = {n: tex[n] for n in fr_methods}

    figure_examples(V, Mte, Cte, seg_preds, methods)
    with open(os.path.join(EVAL_DIR, 'evaluacion_v6.json'), 'w') as fh:
        json.dump(_jsonable(report), fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {EVAL_DIR}/")


if __name__ == '__main__':
    main()
