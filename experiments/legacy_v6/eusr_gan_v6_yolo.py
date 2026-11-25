# ═══════════════════════════════════════════════════════════════
# EUSR-GAN v6 — v5 corregido + YOLO para muestreo guiado por lesión
# Base: Ahmad et al. (2022), Sci. Rep. 12:9533 — adaptado a BUSI
# (ultrasonido mamario), NO a CAMUS como en el paper original.
#
# Cambios respecto a v5:
#   ✓ Split por imagen ANTES de balancear/aumentar (sin fuga de datos)
#   ✓ Eliminación de duplicados de BUSI (hash perceptual) antes del split
#   ✓ Imagen fuente 256×256 → patches HR 128×128 → LR 32×32
#     (en v5 el patch era la imagen completa y el RG no seleccionaba nada)
#   ✓ YOLOv8 entrenado SOLO con el split de train (máscaras BUSI → bbox)
#   ✓ Patches centrados en la lesión detectada por YOLO
#   ✓ Fallback: máscara de tejido (RG vectorizado) para normales/sin detección
#   ✓ Val/test con detecciones de YOLO (no máscaras GT) → protocolo realista
#   ✓ evaluate_bicubic corregido (comparaba contra LR)
#   ✓ save_weights/load_weights (evita error de Keras con capas Lambda)
#   ✓ Etapa 2 con precalentamiento de D + label smoothing + D_STEPS
#   ✓ Métricas por clase (benigno / maligno / normal) en test
#
# Arquitectura del generador, discriminador y losses: idénticos a v5.
# Requisitos: pip install ultralytics scipy scikit-learn
# ═══════════════════════════════════════════════════════════════

import os
import glob
import json
import time
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from scipy import ndimage
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, GroupNormalization, ReLU, LeakyReLU,
    Add, Multiply, Concatenate, Input, Dense,
    GlobalAveragePooling2D, Lambda, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ── Datos ─────────────────────────────────────────────────────
DATASET   = 'busi/Dataset_BUSI_with_GT'
OUT_DIR   = 'resultados_v6_yolo'
DEDUP     = True          # eliminar duplicados conocidos de BUSI

# ── Resolución ────────────────────────────────────────────────
SCALE     = 4
SRC_SIZE  = 256           # imagen fuente (donde YOLO detecta)
HR_PATCH  = 128           # patch HR (ground truth)
LR_PATCH  = HR_PATCH // SCALE   # 32 — entrada del generador

# ── Arquitectura (igual que v5) ───────────────────────────────
BATCH_SIZE = 4           # 16 da solo ~1.3× más muestras/s en CPU y cambia la dinámica
LR_S1      = 5e-5        # si se usa BATCH_SIZE=16, subir a 1e-4 (escalado √4)
EARLY_STOP_S1 = 16       # fin de Etapa 1 si no mejora SSIM en 16 épocas
GN_GROUPS  = 8
NUM_RRDB   = 4
GROWTH_CH  = 32
CH         = 64

# ── YOLO ──────────────────────────────────────────────────────
YOLO_WEIGHTS = 'yolov8n.pt'   # nano: el más liviano para CPU
YOLO_EPOCHS  = 50
YOLO_BATCH   = 16
YOLO_CONF    = 0.25
YOLO_DIR     = 'busi_yolo'
YOLO_REUSE   = True           # si ya existe best.pt, no reentrenar

# ── Muestreo de patches ───────────────────────────────────────
PATCHES_PER_IMG = 3      # patches por imagen en train
JITTER          = 24     # desplazamiento aleatorio del centro (px, en 256)
RG_THRESHOLD    = 0.15   # fallback de tejido
RG_MIN_COVERAGE = 0.30
RG_TRIES        = 20

# ── Entrenamiento ─────────────────────────────────────────────
EPOCHS_S1 = 50
EPOCHS_S2 = 30

os.makedirs(OUT_DIR, exist_ok=True)
W_S1 = os.path.join(OUT_DIR, 'g_v6_stage1_best.weights.h5')
W_S2 = os.path.join(OUT_DIR, 'g_v6_stage2_best.weights.h5')


# ═══════════════════════════════════════════════════════════════
# 1. INVENTARIO, DUPLICADOS Y SPLIT POR IMAGEN
# ═══════════════════════════════════════════════════════════════
def list_busi(path: str) -> list:
    """Lista imágenes BUSI con todas sus máscaras (_mask, _mask_1, ...)."""
    items = []
    for clase in sorted(os.listdir(path)):
        cp = os.path.join(path, clase)
        if not os.path.isdir(cp):
            continue
        for f in sorted(os.listdir(cp)):
            fl = f.lower()
            if '_mask' in fl or not fl.endswith('.png'):
                continue
            stem  = f[:-4]
            masks = sorted(glob.glob(
                os.path.join(cp, glob.escape(stem) + '_mask*.png')))
            items.append({'img': os.path.join(cp, f), 'masks': masks,
                          'clase': clase, 'id': f'{clase}/{stem}'})
    return items


DEDUP_THUMB = 64      # miniatura para comparar imágenes
DEDUP_MAD   = 0.008   # diferencia media absoluta máxima para ser duplicado


def _thumb(fp: str, n: int = DEDUP_THUMB) -> np.ndarray:
    a = np.asarray(Image.open(fp).convert('L').resize((n, n), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    return a.ravel()


def dedupe(items: list, thr: float = DEDUP_MAD) -> list:
    """
    Elimina duplicados casi exactos: dos imágenes son duplicadas si sus
    miniaturas 64×64 difieren en menos de `thr` en promedio (≈2 niveles
    de gris). Criterio conservador: solo descarta copias reales, no
    imágenes parecidas. Imprime los pares para revisarlos a mano.
    """
    T = np.stack([_thumb(it['img']) for it in items])
    keep, dups = np.ones(len(items), dtype=bool), []
    for i in range(len(items)):
        if not keep[i]:
            continue
        rest = np.arange(i + 1, len(items))
        rest = rest[keep[rest]]
        if len(rest) == 0:
            continue
        mad = np.abs(T[rest] - T[i]).mean(axis=1)
        for j in rest[mad < thr]:
            keep[j] = False
            dups.append((items[j]['id'], items[i]['id'],
                         float(mad[rest == j][0])))
    print(f"Duplicados eliminados: {len(dups)}")
    for d, o, m in dups[:15]:
        print(f"  {d}  ≈  {o}   (MAD={m:.4f})")
    if len(dups) > 15:
        print(f"  ... y {len(dups) - 15} más")
    return [it for it, k in zip(items, keep) if k]


def split_items(items: list, seed: int = SEED,
                frac_val: float = 0.1, frac_test: float = 0.1):
    """Split 80/10/10 estratificado por clase, a nivel de IMAGEN."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for clase in sorted({it['clase'] for it in items}):
        grp = [it for it in items if it['clase'] == clase]
        grp = [grp[k] for k in rng.permutation(len(grp))]
        n   = len(grp)
        n_te = max(1, round(n * frac_test)) if n >= 3 else 0
        n_va = max(1, round(n * frac_val))  if n >= 3 else 0
        te += grp[:n_te]
        va += grp[n_te:n_te + n_va]
        tr += grp[n_te + n_va:]
        print(f"  {clase:10s}: train={n - n_te - n_va:3d}  val={n_va:3d}  "
              f"test={n_te:3d}")
    return tr, va, te


# ═══════════════════════════════════════════════════════════════
# 2. CARGA DE IMÁGENES Y MÁSCARAS
# ═══════════════════════════════════════════════════════════════
def load_gray(fp: str, size: int = SRC_SIZE) -> np.ndarray:
    im = Image.open(fp).convert('L').resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(mask_paths: list, size: int = SRC_SIZE) -> np.ndarray:
    """Unión de todas las máscaras de la imagen (lesiones múltiples)."""
    m = np.zeros((size, size), dtype=bool)
    for mp in mask_paths:
        mm = Image.open(mp).convert('L').resize((size, size), Image.NEAREST)
        m |= np.asarray(mm) > 127
    return m


def tissue_mask(img: np.ndarray, thr: float = RG_THRESHOLD) -> np.ndarray:
    """
    Region growing vectorizado (rápido). Fallback cuando YOLO no
    detecta lesión: imágenes normales o lesiones no detectadas.
    """
    lo, hi = np.percentile(img, 20), np.percentile(img, 85)
    m      = (img > lo) & (img < hi + 0.1)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(m)
    H, W = img.shape
    c    = lab[H // 2, W // 2]
    if c == 0:
        cnt = np.bincount(lab[H//3:2*H//3, W//3:2*W//3].ravel())
        cnt[0] = 0
        if cnt.max() > 0:
            c = int(cnt.argmax())
        else:
            c = int(np.bincount(lab.ravel())[1:].argmax()) + 1
    m    = lab == c
    ring = ndimage.binary_dilation(m, structure=np.ones((5, 5))) & ~m
    m   |= ring & (np.abs(img - img[m].mean()) <= 2 * thr)
    m    = ndimage.binary_closing(m, structure=np.ones((9, 9)))
    m    = ndimage.binary_opening(m, structure=np.ones((5, 5)))
    return ndimage.binary_fill_holes(m)


# ═══════════════════════════════════════════════════════════════
# 3. YOLO — DATASET, ENTRENAMIENTO E INFERENCIA
# ═══════════════════════════════════════════════════════════════
def mask_to_yolo_lines(mask: np.ndarray, min_area: int = 20) -> list:
    """Una bbox YOLO por componente conexa de la máscara (clase 0 = lesión)."""
    lab, _ = ndimage.label(mask)
    H, W   = mask.shape
    lines  = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        h, w   = ys.stop - ys.start, xs.stop - xs.start
        if h * w < min_area:
            continue
        cx = (xs.start + xs.stop) / 2 / W
        cy = (ys.start + ys.stop) / 2 / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def prepare_yolo_dataset(splits: dict, cache: dict,
                         out: str = YOLO_DIR) -> str:
    """
    Escribe el dataset YOLO con el MISMO split que la SR.
    YOLO entrena con train, selecciona best.pt con val y se
    reporta en test. Las imágenes normales quedan sin etiquetas
    (ejemplos de fondo).
    """
    if os.path.exists(out):
        shutil.rmtree(out)
    for split, items in splits.items():
        os.makedirs(f'{out}/images/{split}', exist_ok=True)
        os.makedirs(f'{out}/labels/{split}', exist_ok=True)
        for it in items:
            name = it['id'].replace('/', '__').replace(' ', '_')
            Image.fromarray((cache[it['id']] * 255).astype(np.uint8)).save(
                f'{out}/images/{split}/{name}.png')
            lines = mask_to_yolo_lines(load_mask(it['masks']))
            with open(f'{out}/labels/{split}/{name}.txt', 'w') as fh:
                fh.write('\n'.join(lines))
    yaml_path = os.path.join(out, 'data.yaml')
    with open(yaml_path, 'w') as fh:
        fh.write(f"path: {os.path.abspath(out)}\n"
                 "train: images/train\nval: images/val\ntest: images/test\n"
                 "nc: 1\nnames: ['lesion']\n")
    print(f"Dataset YOLO listo: {yaml_path}")
    return yaml_path


def train_yolo(data_yaml: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'ultralytics', '-q'], check=True)
        from ultralytics import YOLO

    project = os.path.abspath('yolo_busi')   # ruta absoluta: ultralytics
    best = os.path.join(project, 'lesion', 'weights', 'best.pt')  # no la mueve
    if YOLO_REUSE and os.path.exists(best):
        print(f"Reutilizando detector: {best}")
        return YOLO(best)

    model = YOLO(YOLO_WEIGHTS)
    model.train(data=data_yaml, epochs=YOLO_EPOCHS, imgsz=SRC_SIZE,
                batch=YOLO_BATCH, device='cpu', project=project,
                name='lesion', exist_ok=True, seed=SEED, workers=4,
                fliplr=0.5, verbose=False)
    best = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    print(f"Detector entrenado: {best}")
    return YOLO(best)


def report_yolo(model, data_yaml: str) -> dict:
    """mAP del detector en el split de test (para la tesis)."""
    try:
        m = model.val(data=data_yaml, split='test', imgsz=SRC_SIZE,
                      device='cpu', verbose=False)
        res = {'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map),
               'precision': float(m.box.mp), 'recall': float(m.box.mr)}
        print("YOLO en test → " + "  ".join(f"{k}: {v:.3f}"
                                             for k, v in res.items()))
        return res
    except Exception as e:
        print(f"No se pudo evaluar YOLO: {e}")
        return {}


def detect_all(model, items: list, cache: dict,
               conf: float = YOLO_CONF) -> dict:
    """Detecciones (x1, y1, x2, y2, conf) ordenadas por confianza."""
    dets = {}
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        arrs  = [np.repeat((cache[it['id']] * 255).astype(np.uint8)[..., None],
                           3, axis=-1) for it in chunk]
        res = model.predict(source=arrs, imgsz=SRC_SIZE, conf=conf,
                            device='cpu', verbose=False)
        for it, r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                dets[it['id']] = []
                continue
            xyxy  = b.xyxy.cpu().numpy()
            cf    = b.conf.cpu().numpy()
            order = np.argsort(-cf)
            dets[it['id']] = [(*xyxy[k].tolist(), float(cf[k])) for k in order]
    return dets


def detection_stats(items: list, dets: dict, nombre: str):
    print(f"  {nombre}:")
    for clase in sorted({it['clase'] for it in items}):
        grp = [it for it in items if it['clase'] == clase]
        con = sum(1 for it in grp if dets[it['id']])
        print(f"    {clase:10s}: {con}/{len(grp)} con detección "
              f"({con / max(1, len(grp)):.0%})")


# ═══════════════════════════════════════════════════════════════
# 4. MUESTREO DE PATCHES GUIADO POR YOLO
# ═══════════════════════════════════════════════════════════════
_mask_cache = {}


def _tissue(item_id: str, img: np.ndarray) -> np.ndarray:
    if item_id not in _mask_cache:
        _mask_cache[item_id] = tissue_mask(img)
    return _mask_cache[item_id]


def pick_center(item_id, img, boxes, rng, train: bool):
    """
    Centro del patch:
      1. Lesión detectada por YOLO (train: con jitter y elección
         ponderada por confianza; val/test: la de mayor confianza)
      2. Fallback: tejido (region growing vectorizado)
      3. Fallback final: centro de la imagen
    """
    H, W = img.shape
    if boxes:
        if train:
            cf = np.array([b[4] for b in boxes])
            k  = int(rng.choice(len(boxes), p=cf / cf.sum()))
        else:
            k = 0
        x1, y1, x2, y2, _ = boxes[k]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if train:
            cx += rng.uniform(-JITTER, JITTER)
            cy += rng.uniform(-JITTER, JITTER)
        return cx, cy, 'yolo'

    mask = _tissue(item_id, img)
    if mask.any():
        if train:
            for _ in range(RG_TRIES):
                top  = int(rng.integers(0, H - HR_PATCH + 1))
                left = int(rng.integers(0, W - HR_PATCH + 1))
                if (mask[top:top + HR_PATCH, left:left + HR_PATCH].mean()
                        >= RG_MIN_COVERAGE):
                    return left + HR_PATCH / 2, top + HR_PATCH / 2, 'tejido'
        ys, xs = np.nonzero(mask)
        return xs.mean(), ys.mean(), 'tejido'
    return W / 2, H / 2, 'centro'


def extract_patch(img, cx, cy, size: int = HR_PATCH):
    H, W = img.shape
    top  = int(np.clip(round(cy - size / 2), 0, H - size))
    left = int(np.clip(round(cx - size / 2), 0, W - size))
    return img[top:top + size, left:left + size].copy(), top, left


def degrade(hr: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """LR por downsampling bicúbico ×4 (protocolo de Ahmad et al.)."""
    im = Image.fromarray(np.clip(hr * 255, 0, 255).astype(np.uint8))
    lr = im.resize((hr.shape[1] // scale, hr.shape[0] // scale), Image.BICUBIC)
    return np.asarray(lr, dtype=np.float32) / 255.0


def augment(hr: np.ndarray, rng) -> np.ndarray:
    if rng.random() > 0.5:
        hr = hr[:, ::-1]
    if rng.random() > 0.5:
        hr = hr[::-1, :]
    if rng.random() > 0.5:
        hr = np.rot90(hr, k=int(rng.integers(1, 4)))
    return np.ascontiguousarray(hr)


def build_split(items, dets, cache, train: bool, seed: int = SEED):
    """
    train: oversample por clase (solo aquí) + PATCHES_PER_IMG patches
           por imagen con jitter y augmentation.
    val/test: un patch determinista por imagen, centrado en la
              detección de YOLO de mayor confianza.
    """
    rng = np.random.default_rng(seed)

    if train:
        by_cls = {}
        for it in items:
            by_cls.setdefault(it['clase'], []).append(it)
        max_n, items_bal = max(len(v) for v in by_cls.values()), []
        for clase, grp in sorted(by_cls.items()):
            extra = [grp[i] for i in rng.choice(len(grp), max_n - len(grp),
                                                replace=True)] \
                if len(grp) < max_n else []
            items_bal += grp + extra
            print(f"    {clase:10s}: {len(grp)} → {len(grp) + len(extra)} imágenes")
        items, n_per = items_bal, PATCHES_PER_IMG
    else:
        n_per = 1

    hr_l, lr_l, cls_l, meta = [], [], [], []
    fuentes = {'yolo': 0, 'tejido': 0, 'centro': 0}
    for it in items:
        img = cache[it['id']]
        for _ in range(n_per):
            cx, cy, src = pick_center(it['id'], img, dets[it['id']],
                                      rng, train)
            hr, top, left = extract_patch(img, cx, cy)
            if train:
                hr = augment(hr, rng)
            hr_l.append(hr[..., None])
            lr_l.append(degrade(hr)[..., None])
            cls_l.append(it['clase'])
            meta.append({'id': it['id'], 'top': top, 'left': left,
                         'src': src})
            fuentes[src] += 1

    hr_a = np.array(hr_l, dtype=np.float32)
    lr_a = np.array(lr_l, dtype=np.float32)
    cls_a = np.array(cls_l)
    if train:
        p = rng.permutation(len(hr_a))
        hr_a, lr_a, cls_a = hr_a[p], lr_a[p], cls_a[p]
        meta = [meta[i] for i in p]
    print(f"    patches={len(hr_a)}  origen → " +
          "  ".join(f"{k}: {v}" for k, v in fuentes.items()))
    return lr_a, hr_a, cls_a, meta


# ═══════════════════════════════════════════════════════════════
# 5. BLOQUES BASE — idénticos a v5
# ═══════════════════════════════════════════════════════════════
def gn_relu(x, groups: int = GN_GROUPS):
    g = min(groups, x.shape[-1])
    while x.shape[-1] % g != 0:
        g //= 2
    return ReLU()(GroupNormalization(groups=g)(x))


def depthwise_sep_conv(x, filters: int, kernel_size: int):
    x = DepthwiseConv2D(kernel_size, padding='same', use_bias=False)(x)
    return Conv2D(filters, 1, use_bias=False)(x)


def dilated_conv_block(x, filters: int, dilation: int = 3):
    x = Conv2D(filters, 3, padding='same',
               dilation_rate=dilation, use_bias=False)(x)
    return gn_relu(x)


def se_block(x, reduction: int = 16):
    ch  = x.shape[-1]
    mid = max(ch // reduction, 8)
    s   = GlobalAveragePooling2D()(x)
    s   = Dense(mid, activation='relu')(s)
    s   = Dense(ch, activation='sigmoid')(s)
    s   = Reshape((1, 1, ch))(s)
    return Multiply()([x, s])


def pixel_shuffle(x, scale: int = 2):
    ch = x.shape[-1]
    x  = Conv2D(ch * scale * scale, 3, padding='same', use_bias=False)(x)
    return Lambda(lambda t: tf.nn.depth_to_space(t, scale))(x)


def dense_block(x, growth: int = GROWTH_CH, n_layers: int = 4):
    in_ch = x.shape[-1]
    feats = [x]
    for _ in range(n_layers):
        h = Concatenate()(feats)
        h = Conv2D(growth, 3, padding='same', use_bias=False)(h)
        h = gn_relu(h)
        feats.append(h)
    out = Concatenate()(feats)
    return Conv2D(in_ch, 1, use_bias=False)(out)


def rrdb_block(x, beta: float = 0.2):
    res = x
    x   = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    x   = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    x   = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    return Add()([res, Lambda(lambda t: t * beta)(x)])


# ═══════════════════════════════════════════════════════════════
# 6. GENERADOR — idéntico a v5 (entrada LR 32×32)
# ═══════════════════════════════════════════════════════════════
def build_generator(lr_size: int = LR_PATCH, n_rrdb: int = NUM_RRDB,
                    ch: int = CH):
    inp = Input(shape=(lr_size, lr_size, 1), name='lr_input')

    # Parte 1 — extracción multi-escala (K3 / K5 separable / K7 dilatada)
    stem = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(inp))

    b3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(stem))
    b3 = GroupNormalization(groups=GN_GROUPS)(
             Conv2D(ch, 3, padding='same', use_bias=False)(b3))
    b3 = ReLU()(Add()([b3, stem]))

    b5 = gn_relu(depthwise_sep_conv(stem, ch, 5))
    b5 = GroupNormalization(groups=GN_GROUPS)(depthwise_sep_conv(b5, ch, 5))
    b5 = ReLU()(Add()([b5, stem]))

    b7 = dilated_conv_block(dilated_conv_block(stem, ch, 3), ch, 3)

    p1 = Concatenate(axis=-1, name='p1_concat')([b3, b5, b7])
    p1 = se_block(p1, reduction=16)
    p1 = Add(name='p1_output')([p1, Conv2D(ch * 3, 1, use_bias=False)(inp)])

    # Parte 2 — RRDB trunk + ×2
    p2    = gn_relu(Conv2D(ch, 1, use_bias=False)(p1))
    p2_in = p2
    for _ in range(n_rrdb):
        p2 = rrdb_block(p2)
    p2 = Add()([Conv2D(ch, 3, padding='same', use_bias=False)(p2), p2_in])
    p2 = gn_relu(pixel_shuffle(p2, scale=2))

    # Parte 3 — mini-red + skip denso desde Parte 1 + ×2
    p3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p2))
    p3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p3))
    skip_p1 = Lambda(
        lambda t: tf.image.resize(t, (lr_size * 2, lr_size * 2),
                                  method='bilinear'),
        name='skip_p1_upsample')(Conv2D(ch, 1, use_bias=False)(p1))
    p3  = Add(name='skip_dense_add')([p3, skip_p1])
    p3  = gn_relu(pixel_shuffle(p3, scale=2))
    p3  = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p3))
    out = Conv2D(1, 3, padding='same', activation='sigmoid',
                 name='hr_output')(p3)

    model = Model(inputs=inp, outputs=out, name='EUSR_GAN_v6')
    total = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Generador v6 | parámetros: {total:,}")
    return model


# ═══════════════════════════════════════════════════════════════
# 7. DISCRIMINADOR — idéntico a v5
# ═══════════════════════════════════════════════════════════════
def build_discriminator(hr_size: int = HR_PATCH):
    inp = Input(shape=(hr_size, hr_size, 1))

    def d_block(x, filters, stride=2, norm=True):
        x = Conv2D(filters, 4, strides=stride, padding='same',
                   use_bias=False)(x)
        if norm:
            g = min(GN_GROUPS, filters)
            while filters % g != 0:
                g //= 2
            x = GroupNormalization(groups=g)(x)
        return LeakyReLU(0.2)(x)

    x   = d_block(inp, 64, stride=2, norm=False)
    x   = d_block(x, 128, stride=2)
    x   = d_block(x, 256, stride=2)
    x   = d_block(x, 512, stride=1)
    x   = GlobalAveragePooling2D()(x)
    x   = Dense(1024)(x)
    x   = LeakyReLU(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name='PatchDiscriminator')


def reset_discriminator(D):
    for layer in D.layers:
        if getattr(layer, 'kernel', None) is not None:
            layer.kernel.assign(layer.kernel_initializer(
                layer.kernel.shape, dtype=layer.kernel.dtype))
        if getattr(layer, 'bias', None) is not None:
            layer.bias.assign(layer.bias_initializer(
                layer.bias.shape, dtype=layer.bias.dtype))
    print("Discriminador reiniciado.")


# ═══════════════════════════════════════════════════════════════
# 8. LOSSES — idénticas a v5 (+ label smoothing en D)
# ═══════════════════════════════════════════════════════════════
_vgg_base = VGG19(weights='imagenet', include_top=False,
                  input_shape=(HR_PATCH, HR_PATCH, 3))
vgg_feat = Model(inputs=_vgg_base.input,
                 outputs=_vgg_base.get_layer('block5_conv4').output,
                 name='VGG_features')
vgg_feat.trainable = False


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def perceptual_loss(y_true, y_pred):
    t_rgb = tf.image.grayscale_to_rgb(y_true)
    p_rgb = tf.image.grayscale_to_rgb(y_pred)
    return tf.reduce_mean(tf.square(vgg_feat(t_rgb) - vgg_feat(p_rgb)))


def fft_loss(y_true, y_pred):
    """Preserva la firma espectral del speckle de ultrasonido."""
    mag_t = tf.abs(tf.signal.rfft2d(tf.squeeze(y_true, axis=-1)))
    mag_p = tf.abs(tf.signal.rfft2d(tf.squeeze(y_pred, axis=-1)))
    mag_t = mag_t / (tf.reduce_max(mag_t) + 1e-8)
    mag_p = mag_p / (tf.reduce_max(mag_p) + 1e-8)
    return tf.reduce_mean(tf.abs(mag_t - mag_p))


def combined_generator_loss(y_true, y_pred, disc_pred_fake,
                            lambda_l1=1.0, lambda_ssim=0.1,
                            lambda_freq=0.05, lambda_perc=0.05,
                            lambda_adv=0.001, stage=1):
    l_l1   = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_ssim = ssim_loss(y_true, y_pred)
    l_freq = fft_loss(y_true, y_pred)
    loss   = lambda_l1 * l_l1 + lambda_ssim * l_ssim + lambda_freq * l_freq
    parts  = {'l1': l_l1, 'ssim': l_ssim, 'freq': l_freq}
    if stage == 2:
        l_perc = perceptual_loss(y_true, y_pred)
        loss  += lambda_perc * l_perc
        parts['perc'] = l_perc
        if disc_pred_fake is not None:
            l_adv = tf.reduce_mean(tf.square(disc_pred_fake - 1.0))
            loss += lambda_adv * l_adv
            parts['adv'] = l_adv
    return loss, parts


def discriminator_loss(real_pred, fake_pred, smooth: bool = True):
    """LSGAN con label smoothing (HR → 0.9, SR → 0.1)."""
    rt, ft = (0.9, 0.1) if smooth else (1.0, 0.0)
    return 0.5 * (tf.reduce_mean(tf.square(real_pred - rt)) +
                  tf.reduce_mean(tf.square(fake_pred - ft)))


# ═══════════════════════════════════════════════════════════════
# 9. MÉTRICAS
# ═══════════════════════════════════════════════════════════════
def psnr_per_image(y_true, y_pred) -> np.ndarray:
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    return (10.0 * tf.math.log(1.0 / (mse + 1e-10)) /
            tf.math.log(10.0)).numpy()


def ssim_per_image(y_true, y_pred) -> np.ndarray:
    return tf.image.ssim(y_true, y_pred, max_val=1.0).numpy()


def predict_sr(G, lr, batch: int = 16) -> np.ndarray:
    return np.concatenate([G(lr[i:i + batch], training=False).numpy()
                           for i in range(0, len(lr), batch)])


def bicubic_sr(lr) -> np.ndarray:
    return np.clip(tf.image.resize(lr, (HR_PATCH, HR_PATCH),
                                   method='bicubic').numpy(), 0.0, 1.0)


def evaluate(G, lr, hr):
    sr = predict_sr(G, lr)
    return (float(np.mean(psnr_per_image(hr, sr))),
            float(np.mean(ssim_per_image(hr, sr))))


# ═══════════════════════════════════════════════════════════════
# 10. ETAPA 1 — preentrenamiento supervisado (igual que v5)
# ═══════════════════════════════════════════════════════════════
def train_stage1(G, x_tr, y_tr, x_val, y_val,
                 epochs: int = EPOCHS_S1, batch_size: int = BATCH_SIZE):
    print("═" * 60)
    print("ETAPA 1 — Preentrenamiento (L1 + SSIM + FFT)")
    print("═" * 60)

    LR_INIT, LR_MIN, PATIENCE = LR_S1, 1e-5, 8
    current_lr = LR_INIT
    opt        = Adam(current_lr, beta_1=0.9)
    best_ssim, no_improve, sin_mejora_total = -1.0, 0, 0
    n_batches  = max(1, len(x_tr) // batch_size)
    hist       = {'loss': [], 'psnr': [], 'ssim': [], 'freq': []}
    print(f"Train: {len(x_tr)}  Val: {len(x_val)}  Batches/época: {n_batches}\n")

    @tf.function
    def step(lr_b, hr_b):
        with tf.GradientTape() as tape:
            sr = G(lr_b, training=True)
            loss, parts = combined_generator_loss(hr_b, sr, None, stage=1)
        grads, _ = tf.clip_by_global_norm(
            tape.gradient(loss, G.trainable_variables), 1.0)
        opt.apply_gradients(zip(grads, G.trainable_variables))
        return loss, parts['freq']

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        losses, freqs = [], []
        idx = np.random.permutation(len(x_tr))
        for i in range(n_batches):
            b = idx[i * batch_size:(i + 1) * batch_size]
            loss, fq = step(x_tr[b], y_tr[b])
            losses.append(loss.numpy()); freqs.append(fq.numpy())

        psnr, ssim = evaluate(G, x_val, y_val)
        hist['loss'].append(float(np.mean(losses)))
        hist['psnr'].append(psnr); hist['ssim'].append(ssim)
        hist['freq'].append(float(np.mean(freqs)))

        mejora = ssim > best_ssim
        if mejora:
            best_ssim, no_improve, sin_mejora_total = ssim, 0, 0
            G.save_weights(W_S1)
        else:
            no_improve += 1
            sin_mejora_total += 1
        if no_improve >= PATIENCE and current_lr > LR_MIN:
            current_lr = max(current_lr * 0.5, LR_MIN)
            no_improve = 0
            opt.learning_rate.assign(current_lr)
            print(f"  ↓ lr → {current_lr:.2e}")

        print(f"  Época {epoch:03d}/{epochs} | loss: {np.mean(losses):.4f} | "
              f"freq: {np.mean(freqs):.4f} | PSNR: {psnr:.2f} dB | "
              f"SSIM: {ssim:.4f} | {time.time() - t0:.0f}s"
              + (" ← guardado" if mejora else ""), flush=True)

        if sin_mejora_total >= EARLY_STOP_S1:
            print(f"\n  Sin mejora en {EARLY_STOP_S1} épocas — fin de Etapa 1.")
            break

    G.load_weights(W_S1)
    print(f"\nEtapa 1 completa. Mejor SSIM val: {best_ssim:.4f}\n")
    return G, hist, best_ssim


# ═══════════════════════════════════════════════════════════════
# 11. ETAPA 2 — GAN con precalentamiento de D y label smoothing
# ═══════════════════════════════════════════════════════════════
def train_stage2(G, D, x_tr, y_tr, x_val, y_val, best_s1: float,
                 epochs: int = EPOCHS_S2, batch_size: int = BATCH_SIZE):
    print("═" * 60)
    print("ETAPA 2 — Fine-tuning GAN (precalentamiento D + smoothing)")
    print("═" * 60)

    reset_discriminator(D)
    n_batches = max(1, len(x_tr) // batch_size)

    # ── Precalentamiento: D aprende con G congelado ──────────────
    opt_dw = Adam(1e-4, beta_1=0.9)

    @tf.function
    def step_warm(lr_b, hr_b):
        with tf.GradientTape() as td:
            sr = G(lr_b, training=False)
            rp = D(hr_b, training=True)
            fp = D(sr, training=True)
            ld = discriminator_loss(rp, fp)
        gd, _ = tf.clip_by_global_norm(td.gradient(ld, D.trainable_variables), 1.0)
        opt_dw.apply_gradients(zip(gd, D.trainable_variables))
        return ld, tf.reduce_mean(fp)

    for ep in range(1, 4):
        dl, fpm = [], []
        idx = np.random.permutation(len(x_tr))
        for i in range(n_batches):
            b = idx[i * batch_size:(i + 1) * batch_size]
            ld, fm = step_warm(x_tr[b], y_tr[b])
            dl.append(ld.numpy()); fpm.append(fm.numpy())
        print(f"  Warmup {ep}/3 | D: {np.mean(dl):.4f} | fake→D: {np.mean(fpm):.3f}")

    use_gan = float(np.mean(fpm)) < 0.45
    print(f"\n{'✓ D distingue SR de HR — modo GAN' if use_gan else '⚠ D no distingue — modo perceptual (sin adversarial)'}\n")

    # ── Entrenamiento principal ─────────────────────────────────
    D_STEPS, PATIENCE = 3, 8
    opt_g = Adam(1e-5, beta_1=0.9)
    opt_d = Adam(2e-5, beta_1=0.9)
    best_s2, no_improve, saved = best_s1, 0, False
    hist = {'loss_G': [], 'loss_D': [], 'psnr': [], 'ssim': []}

    @tf.function
    def step_d(lr_b, hr_b):
        with tf.GradientTape() as td:
            sr = G(lr_b, training=False)
            ld = discriminator_loss(D(hr_b, training=True),
                                    D(sr, training=True))
        gd, _ = tf.clip_by_global_norm(td.gradient(ld, D.trainable_variables), 1.0)
        opt_d.apply_gradients(zip(gd, D.trainable_variables))
        return ld

    @tf.function
    def step_g_gan(lr_b, hr_b):
        with tf.GradientTape() as tg:
            sr = G(lr_b, training=True)
            lg, _ = combined_generator_loss(hr_b, sr, D(sr, training=False),
                                            lambda_adv=0.0005, stage=2)
        gg, _ = tf.clip_by_global_norm(tg.gradient(lg, G.trainable_variables), 1.0)
        opt_g.apply_gradients(zip(gg, G.trainable_variables))
        return lg

    @tf.function
    def step_g_perc(lr_b, hr_b):
        with tf.GradientTape() as tg:
            sr = G(lr_b, training=True)
            lg, _ = combined_generator_loss(hr_b, sr, None, stage=2)
        gg, _ = tf.clip_by_global_norm(tg.gradient(lg, G.trainable_variables), 1.0)
        opt_g.apply_gradients(zip(gg, G.trainable_variables))
        return lg

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        gl, dl = [], []
        idx = np.random.permutation(len(x_tr))
        for i in range(n_batches):
            b = idx[i * batch_size:(i + 1) * batch_size]
            if use_gan:
                dl.append(np.mean([step_d(x_tr[b], y_tr[b]).numpy()
                                   for _ in range(D_STEPS)]))
                gl.append(step_g_gan(x_tr[b], y_tr[b]).numpy())
            else:
                gl.append(step_g_perc(x_tr[b], y_tr[b]).numpy())

        psnr, ssim = evaluate(G, x_val, y_val)
        d_mean = float(np.mean(dl)) if dl else 0.0
        hist['loss_G'].append(float(np.mean(gl))); hist['loss_D'].append(d_mean)
        hist['psnr'].append(psnr); hist['ssim'].append(ssim)

        mejora = ssim > best_s2
        if mejora:
            best_s2, no_improve, saved = ssim, 0, True
            G.save_weights(W_S2)
        else:
            no_improve += 1

        estado = ''
        if use_gan and d_mean < 0.05:
            estado = '  ⚠ D bajo'
        elif use_gan and d_mean > 0.45:
            estado = '  ⚠ D alto'
        print(f"  [{'GAN' if use_gan else 'Perc'}] Época {epoch:03d}/{epochs} | "
              f"G: {np.mean(gl):.4f} | D: {d_mean:.4f} | "
              f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | {time.time() - t0:.0f}s"
              + (" ← guardado" if mejora else "") + estado, flush=True)
        if no_improve >= PATIENCE:
            print(f"\n  Sin mejora en {PATIENCE} épocas — fin de Etapa 2.")
            break

    if saved:
        G.load_weights(W_S2)
        print(f"\nEtapa 2 mejoró la Etapa 1 → SSIM val {best_s2:.4f}")
    else:
        G.load_weights(W_S1)
        print("\nEtapa 2 no superó la Etapa 1 → se conservan pesos de Etapa 1")
    return G, hist, saved


# ═══════════════════════════════════════════════════════════════
# 12. FIGURAS
# ═══════════════════════════════════════════════════════════════
def plot_curves(hist_s1: dict, hist_s2: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(hist_s1['loss'], label='S1', color='steelblue')
    axes[0].plot(hist_s2['loss_G'], label='S2-G', color='coral')
    if any(hist_s2['loss_D']):
        axes[0].plot(hist_s2['loss_D'], label='S2-D', color='seagreen', ls='--')
    axes[0].set_title('Pérdidas')
    for k, ax, t in [('psnr', axes[1], 'PSNR val (dB)'),
                     ('ssim', axes[2], 'SSIM val')]:
        ax.plot(hist_s1[k], 'o-', ms=3, label='S1', color='steelblue')
        ax.plot(range(len(hist_s1[k]), len(hist_s1[k]) + len(hist_s2[k])),
                hist_s2[k], 's-', ms=3, label='S2', color='coral')
        ax.set_title(t)
    for ax in axes:
        ax.set_xlabel('Época'); ax.legend(); ax.grid(alpha=.3)
    plt.suptitle('EUSR-GAN v6 (YOLO-guided) — BUSI', fontsize=13)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'training_curves_v6.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Curvas guardadas: {fp}")


def visualize_results(G, lr, hr, cls, meta, cache, dets, n: int = 4):
    """Fuente con bbox YOLO y ventana del patch | LR | Bicúbica | SR | HR."""
    elegidos = []
    for clase in ['benign', 'malignant', 'normal']:
        elegidos += [i for i, c in enumerate(cls) if c == clase][:max(1, n // 3)]
    elegidos = elegidos[:n]

    sr_all = predict_sr(G, lr[elegidos])
    bic    = bicubic_sr(lr[elegidos])
    fig, axes = plt.subplots(len(elegidos), 5, figsize=(17, 3.6 * len(elegidos)))
    axes = np.atleast_2d(axes)
    titulos = ['Fuente 256 + YOLO', f'LR {LR_PATCH}×{LR_PATCH}',
               'Bicúbica', 'EUSR-GAN v6', f'HR {HR_PATCH}×{HR_PATCH}']

    for r, i in enumerate(elegidos):
        m   = meta[i]
        src = cache[m['id']]
        axes[r, 0].imshow(src, cmap='gray', vmin=0, vmax=1)
        for (x1, y1, x2, y2, cf) in dets[m['id']][:1]:
            axes[r, 0].add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, ec='red', lw=1.5))
            axes[r, 0].text(x1, y1 - 3, f'{cf:.2f}', color='red', fontsize=8)
        axes[r, 0].add_patch(mpatches.Rectangle(
            (m['left'], m['top']), HR_PATCH, HR_PATCH,
            fill=False, ec='lime', lw=1.2, ls='--'))
        axes[r, 0].set_ylabel(f"{cls[i]} ({m['src']})", fontsize=9)

        for c, img in enumerate([lr[i], bic[r], sr_all[r], hr[i]], start=1):
            axes[r, c].imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        for c, img in [(2, bic[r]), (3, sr_all[r])]:
            p = psnr_per_image(hr[i:i + 1], img[None])[0]
            s = ssim_per_image(hr[i:i + 1], img[None])[0]
            axes[r, c].set_xlabel(f'PSNR {p:.2f} dB  SSIM {s:.3f}', fontsize=9)
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titulos[c], fontsize=10)

    plt.suptitle('Rojo: detección YOLO · Verde: ventana del patch', fontsize=11)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'sr_results_v6.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.show()
    print(f"Figura guardada: {fp}")


# ═══════════════════════════════════════════════════════════════
# 13. EVALUACIÓN FINAL EN TEST (global y por clase)
# ═══════════════════════════════════════════════════════════════
def final_report(G, lr, hr, cls, yolo_metrics: dict, stage2_used: bool):
    sr  = predict_sr(G, lr)
    bic = bicubic_sr(lr)
    p_sr, s_sr = psnr_per_image(hr, sr), ssim_per_image(hr, sr)
    p_bc, s_bc = psnr_per_image(hr, bic), ssim_per_image(hr, bic)

    def fila(nombre, p, s):
        return (f"{nombre:<26} {p.mean():6.2f} ± {p.std():4.2f} dB   "
                f"{s.mean():.4f} ± {s.std():.4f}")

    print("\n" + "═" * 70)
    print("TABLA — Test BUSI (patches 128×128 centrados por YOLO, ×4)")
    print("═" * 70)
    print(f"{'Método':<26} {'PSNR':>17}   {'SSIM':>15}")
    print("─" * 70)
    print(fila('Bicúbica', p_bc, s_bc))
    print(fila('EUSR-GAN v6 (YOLO)', p_sr, s_sr))
    print("─" * 70)
    print("Por clase (EUSR-GAN v6 | Bicúbica):")
    por_clase = {}
    for clase in sorted(set(cls)):
        k = cls == clase
        por_clase[clase] = {
            'n': int(k.sum()),
            'psnr_sr': float(p_sr[k].mean()), 'ssim_sr': float(s_sr[k].mean()),
            'psnr_bic': float(p_bc[k].mean()), 'ssim_bic': float(s_bc[k].mean())}
        print(f"  {clase:10s} n={k.sum():3d} | "
              f"{p_sr[k].mean():.2f} dB / {s_sr[k].mean():.4f} | "
              f"{p_bc[k].mean():.2f} dB / {s_bc[k].mean():.4f}")
    print("═" * 70)

    res = {
        'config': {'SRC_SIZE': SRC_SIZE, 'HR_PATCH': HR_PATCH, 'SCALE': SCALE,
                   'NUM_RRDB': NUM_RRDB, 'GROWTH_CH': GROWTH_CH, 'CH': CH,
                   'PATCHES_PER_IMG': PATCHES_PER_IMG, 'JITTER': JITTER,
                   'YOLO_CONF': YOLO_CONF, 'DEDUP': DEDUP,
                   'stage2_used': stage2_used},
        'yolo_test': yolo_metrics,
        'test_global': {
            'bicubic': {'psnr': float(p_bc.mean()), 'psnr_std': float(p_bc.std()),
                        'ssim': float(s_bc.mean()), 'ssim_std': float(s_bc.std())},
            'eusr_gan_v6': {'psnr': float(p_sr.mean()), 'psnr_std': float(p_sr.std()),
                            'ssim': float(s_sr.mean()), 'ssim_std': float(s_sr.std())}},
        'test_por_clase': por_clase}
    fp = os.path.join(OUT_DIR, 'resultados_test_v6.json')
    with open(fp, 'w') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"Resultados guardados: {fp}")
    return res


# ═══════════════════════════════════════════════════════════════
# 14. PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════
def main():
    # ── 1. Inventario, duplicados y split por imagen ────────────
    items = list_busi(DATASET)
    print(f"Imágenes BUSI encontradas: {len(items)}")
    if DEDUP:
        items = dedupe(items)
    print("Split por imagen (estratificado):")
    tr_items, va_items, te_items = split_items(items)

    cache = {it['id']: load_gray(it['img']) for it in items}

    # ── 2. YOLO: dataset con el mismo split, entrenamiento, test ─
    yaml_path = prepare_yolo_dataset(
        {'train': tr_items, 'val': va_items, 'test': te_items}, cache)
    yolo = train_yolo(yaml_path)
    yolo_metrics = report_yolo(yolo, yaml_path)

    # ── 3. Detecciones (predicciones de YOLO, nunca máscaras GT) ─
    dets = detect_all(yolo, items, cache)
    print("Detecciones por split:")
    detection_stats(tr_items, dets, 'train')
    detection_stats(va_items, dets, 'val')
    detection_stats(te_items, dets, 'test')

    # ── 4. Patches HR/LR guiados por YOLO ───────────────────────
    print("\nConstruyendo patches:")
    print("  train (balanceado + aumentado):")
    x_tr, y_tr, _, _ = build_split(tr_items, dets, cache, train=True)
    print("  val:")
    x_val, y_val, _, _ = build_split(va_items, dets, cache, train=False)
    print("  test:")
    x_te, y_te, c_te, m_te = build_split(te_items, dets, cache, train=False)

    # ── 5. Modelos y entrenamiento ──────────────────────────────
    G = build_generator()
    D = build_discriminator()
    G, hist_s1, best_s1 = train_stage1(G, x_tr, y_tr, x_val, y_val)
    G, hist_s2, s2_used = train_stage2(G, D, x_tr, y_tr, x_val, y_val, best_s1)

    # ── 6. Resultados ───────────────────────────────────────────
    plot_curves(hist_s1, hist_s2)
    visualize_results(G, x_te, y_te, c_te, m_te, cache, dets, n=6)
    final_report(G, x_te, y_te, c_te, yolo_metrics, s2_used)


if __name__ == '__main__':
    main()
