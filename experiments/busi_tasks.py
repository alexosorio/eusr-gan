# ═══════════════════════════════════════════════════════════════
# busi_tasks.py — Evaluación basada en tareas de EUSR-GAN en BUSI
#
# Pregunta: ¿las imágenes reconstruidas conservan la información con la
# que los radiólogos anotaron BUSI? Sin lectores clínicos, se responde
# con dos tareas sobre las anotaciones de BUSI:
#   1. DETECCIÓN de lesiones (YOLOv8n)  → mAP, aciertos por imagen
#   2. SEGMENTACIÓN de lesiones (U-Net) → Dice / IoU por imagen
#   + ALUCINACIONES: detecciones y áreas segmentadas en imágenes NORMALES
#
# Protocolo: los dos modelos se entrenan SOLO con imágenes HR nativas
# de train (y se seleccionan con val). En test se les da la misma imagen
# en varias versiones: HR (techo), LR ampliada por vecino (piso),
# bicúbica (línea base), EUSR-GAN y EUSR-GAN+ (autoensamble ×8).
# Estadística: Wilcoxon pareado + "brecha recuperada" =
# (EUSR − bicúbica) / (HR − bicúbica).
#
# Reutiliza busi_native.py (mismo split, carga nativa, generador).
# Requiere en la carpeta: busi_native.py y los pesos de B1.
# Uso:  python -u busi_tasks.py > log_busi_tasks.txt 2>&1
# ═══════════════════════════════════════════════════════════════

import os
import json
import time
import shutil
import importlib.util

import numpy as np
from PIL import Image
from scipy import ndimage, stats

# ── Configuración ─────────────────────────────────────────────
BASE_FILE = 'busi_native.py'
SR_MODELS = {'EUSR-GAN': 'resultados_busi_native_scratch_yolo/g_busi_stage1_best.weights.h5'}
SELF_ENSEMBLE_TOO = True          # agrega 'EUSR-GAN+' (8× más lenta la inferencia)
OUT_DIR   = 'evaluacion_tareas_busi'

DET_IMGSZ  = 512                  # YOLO a resolución cercana a la nativa (~500 px)
DET_EPOCHS = 50
DET_BATCH  = 8
DET_CONF   = 0.25
YOLO_BASE  = 'yolov8n.pt'

SEG_SIZE     = 384                # U-Net (imagen y máscara reescaladas igual para todas las versiones)
SEG_BASE     = 16
SEG_EPOCHS   = 60
SEG_BATCH    = 4
SEG_PATIENCE = 10
SEG_LR       = 1e-3

SEED = 42


# ═══════════════════════════════════════════════════════════════
# 0. BASE (split, carga nativa y generador idénticos a busi_native)
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(BASE_FILE):
    raise SystemExit(f"Falta {BASE_FILE} en esta carpeta")
_spec = importlib.util.spec_from_file_location('busi_base', BASE_FILE)
bn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bn)
import tensorflow as tf   # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


def slug(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_').lower()


# ═══════════════════════════════════════════════════════════════
# 1. DATOS Y VERSIONES DE TEST
# ═══════════════════════════════════════════════════════════════
def load_data():
    items = bn.list_busi(bn.DATASET)
    if bn.DEDUP:
        items = bn.dedupe(items)
    print("Split (idéntico a v6 / busi_native):")
    tr, va, te = bn.split_items(items)
    return {k: bn.load_set(v) for k, v in (('train', tr), ('val', va), ('test', te))}


def self_ensemble(G, lr):
    acc = None
    for k in range(4):
        for flip in (False, True):
            x = np.rot90(lr, k)
            x = x[:, ::-1] if flip else x
            y = bn.sr_tiles(G, np.ascontiguousarray(x))
            y = y[:, ::-1] if flip else y
            y = np.rot90(y, -k)
            acc = y if acc is None else acc + y
    return acc / 8.0


def build_versions(test):
    """Diccionario versión → lista de imágenes (tamaño nativo de cada una)."""
    V = {'HR': [h.copy() for h in test['hr']]}
    lrs = [bn.degrade(h) for h in test['hr']]
    V['LR (vecino)'] = [np.kron(l, np.ones((bn.SCALE, bn.SCALE), np.float32)) for l in lrs]
    V['Bicúbica'] = [bn.bicubic_full(l, h.shape) for l, h in zip(lrs, test['hr'])]
    for name, w in SR_MODELS.items():
        if not os.path.exists(w):
            raise SystemExit(f"No existen los pesos {w}")
        G = bn.build_generator()
        G.load_weights(w)
        t0 = time.time()
        V[name] = [bn.sr_tiles(G, l) for l in lrs]
        print(f"  {name}: {time.time() - t0:.0f}s")
        if SELF_ENSEMBLE_TOO:
            t0 = time.time()
            V[name + '+'] = [self_ensemble(G, l) for l in lrs]
            print(f"  {name}+: {time.time() - t0:.0f}s")
    return V


# ═══════════════════════════════════════════════════════════════
# 2. ESTADÍSTICA
# ═══════════════════════════════════════════════════════════════
def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    try:
        return float(stats.wilcoxon(a[ok], b[ok]).pvalue) if np.any(a[ok] != b[ok]) else 1.0
    except ValueError:
        return float('nan')


def gap(sr, bic, hr):
    d = hr - bic
    return (sr - bic) / d if abs(d) > 1e-9 else float('nan')


def ms(x, d=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}" if len(x) else 'n/a'


def print_table(title, header, rows):
    print(f"\n{title}")
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print('  ' + ' | '.join(str(h).ljust(w[i]) for i, h in enumerate(header)))
    print('  ' + '-+-'.join('-' * x for x in w))
    for r in rows:
        print('  ' + ' | '.join(str(c).ljust(w[i]) for i, c in enumerate(r)))


# ═══════════════════════════════════════════════════════════════
# 3. DETECCIÓN (YOLOv8n entrenado con HR nativa)
# ═══════════════════════════════════════════════════════════════
def mask_boxes(mask, min_area=20):
    lab, _ = ndimage.label(mask)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) >= min_area:
            out.append((xs.start, ys.start, xs.stop, ys.stop))
    return out


def _u8rgb(img):
    return np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)


def write_yolo_split(root, split, imgs, masks):
    os.makedirs(f'{root}/images/{split}', exist_ok=True)
    os.makedirs(f'{root}/labels/{split}', exist_ok=True)
    for i, (img, m) in enumerate(zip(imgs, masks)):
        Image.fromarray(_u8rgb(img)).save(f'{root}/images/{split}/{i:04d}.png')
        H, W = img.shape
        lines = [f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
                 for x0, y0, x1, y1 in mask_boxes(m)]
        with open(f'{root}/labels/{split}/{i:04d}.txt', 'w') as fh:
            fh.write('\n'.join(lines))


def write_yaml(root, train='images/train', val='images/val', test='images/test'):
    p = f'{root}/data.yaml'
    with open(p, 'w') as fh:
        fh.write(f"path: {root}\ntrain: {train}\nval: {val}\ntest: {test}\nnc: 1\nnames: ['lesion']\n")
    return p


def train_detector(D):
    from ultralytics import YOLO
    project = os.path.abspath(os.path.join(OUT_DIR, 'yolo'))
    best = os.path.join(project, 'det_hr', 'weights', 'best.pt')
    if os.path.exists(best):
        print(f"  Detector reutilizado: {best}")
        return YOLO(best)
    root = os.path.abspath(os.path.join(OUT_DIR, 'yolo_data_hr'))
    shutil.rmtree(root, ignore_errors=True)
    write_yolo_split(root, 'train', D['train']['hr'], D['train']['mask'])
    write_yolo_split(root, 'val', D['val']['hr'], D['val']['mask'])
    write_yolo_split(root, 'test', D['test']['hr'], D['test']['mask'])
    model = YOLO(YOLO_BASE)
    model.train(data=write_yaml(root), epochs=DET_EPOCHS, imgsz=DET_IMGSZ, batch=DET_BATCH,
                device='cpu', project=project, name='det_hr', exist_ok=True, seed=SEED,
                workers=4, fliplr=0.5, plots=False, verbose=False)
    return YOLO(best)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def eval_detection(model, V, test):
    cls = [it['clase'] for it in test['items']]
    res = {}
    for name, imgs in V.items():
        root = os.path.abspath(os.path.join(OUT_DIR, 'yolo_test', slug(name)))
        shutil.rmtree(root, ignore_errors=True)
        write_yolo_split(root, 'test', imgs, test['mask'])
        m = model.val(data=write_yaml(root, 'images/test', 'images/test', 'images/test'),
                      split='test', imgsz=DET_IMGSZ, batch=8, device='cpu', plots=False,
                      verbose=False, project=os.path.abspath(os.path.join(OUT_DIR, 'yolo_val')),
                      name=slug(name), exist_ok=True)
        best_iou, n_fp_normal = [], []
        for k in range(0, len(imgs), 8):
            preds = model.predict([_u8rgb(x) for x in imgs[k:k + 8]], imgsz=DET_IMGSZ,
                                  conf=DET_CONF, device='cpu', verbose=False)
            for j, p in enumerate(preds):
                i = k + j
                boxes = [] if p.boxes is None else p.boxes.xyxy.cpu().numpy().tolist()
                if cls[i] == 'normal':
                    n_fp_normal.append(len(boxes))
                    best_iou.append(np.nan)
                else:
                    gt = mask_boxes(test['mask'][i])
                    best_iou.append(max([box_iou(b, g) for b in boxes for g in gt], default=0.0))
        best_iou = np.array(best_iou)
        res[name] = {'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map),
                     'precision': float(m.box.mp), 'recall': float(m.box.mr),
                     'best_iou': best_iou, 'hit50': (best_iou >= 0.5).astype(float),
                     'fp_normal': np.array(n_fp_normal, float)}
        res[name]['hit50'][np.isnan(best_iou)] = np.nan
    return res


# ═══════════════════════════════════════════════════════════════
# 4. SEGMENTACIÓN (U-Net entrenada con HR nativa)
# ═══════════════════════════════════════════════════════════════
def to_seg(imgs, masks=None):
    X = np.stack([np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
                             .resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR), np.float32) / 255.0
                  for x in imgs])[..., None]
    if masks is None:
        return X
    M = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                             .resize((SEG_SIZE, SEG_SIZE), Image.NEAREST)) > 127
                  for m in masks])[..., None].astype(np.float32)
    return X, M


def build_unet():
    L = tf.keras.layers

    def cbr(x, f):
        for _ in range(2):
            x = L.ReLU()(L.BatchNormalization()(L.Conv2D(f, 3, padding='same', use_bias=False)(x)))
        return x
    inp = L.Input((SEG_SIZE, SEG_SIZE, 1))
    x, skips = inp, []
    for f in (SEG_BASE, SEG_BASE * 2, SEG_BASE * 4, SEG_BASE * 8):
        x = cbr(x, f)
        skips.append(x)
        x = L.MaxPooling2D()(x)
    x = cbr(x, SEG_BASE * 16)
    for f, s in zip((SEG_BASE * 8, SEG_BASE * 4, SEG_BASE * 2, SEG_BASE), reversed(skips)):
        x = L.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = cbr(L.Concatenate()([x, s]), f)
    return tf.keras.Model(inp, L.Conv2D(1, 1, activation='sigmoid')(x))


def dice_coef(y, p, eps=1.0):
    inter = tf.reduce_sum(y * p, axis=[1, 2, 3])
    return tf.reduce_mean((2 * inter + eps) / (tf.reduce_sum(y, axis=[1, 2, 3]) +
                                               tf.reduce_sum(p, axis=[1, 2, 3]) + eps))


def seg_loss(y, p):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, p)) + 1.0 - dice_coef(y, p)


def train_unet(D):
    path = os.path.join(OUT_DIR, 'unet_hr.weights.h5')
    net = build_unet()
    if os.path.exists(path):
        net.load_weights(path)
        print(f"  U-Net reutilizada: {path}")
        return net
    Xtr, Mtr = to_seg(D['train']['hr'], D['train']['mask'])
    Xva, Mva = to_seg(D['val']['hr'], D['val']['mask'])

    def aug(x, m):
        flip = tf.random.uniform(()) > 0.5
        return (tf.where(flip, tf.image.flip_left_right(x), x),
                tf.where(flip, tf.image.flip_left_right(m), m))
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, Mtr)).shuffle(len(Xtr), seed=SEED)
          .map(aug).batch(SEG_BATCH).prefetch(2))
    net.compile(optimizer=tf.keras.optimizers.Adam(SEG_LR), loss=seg_loss, metrics=[dice_coef])
    net.fit(ds, validation_data=(Xva, Mva), epochs=SEG_EPOCHS, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', mode='max',
                                                        patience=SEG_PATIENCE,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', mode='max',
                                                            factor=0.5, patience=4, min_lr=1e-5)])
    net.save_weights(path)
    return net


def eval_segmentation(net, V, test):
    cls = np.array([it['clase'] for it in test['items']])
    _, M = to_seg(test['hr'], test['mask'])
    res = {}
    for name, imgs in V.items():
        P = net.predict(to_seg(imgs), batch_size=8, verbose=0) > 0.5
        G = M > 0.5
        inter = (P & G).sum(axis=(1, 2, 3))
        ps, gs = P.sum(axis=(1, 2, 3)), G.sum(axis=(1, 2, 3))
        union = (P | G).sum(axis=(1, 2, 3))
        les = cls != 'normal'
        dice = np.where(les, 2 * inter / np.maximum(ps + gs, 1), np.nan)
        iou = np.where(les, inter / np.maximum(union, 1), np.nan)
        res[name] = {'dice': dice, 'iou': iou,
                     'fp_area_normal': np.where(~les, ps / (SEG_SIZE * SEG_SIZE), np.nan)}
    return res


# ═══════════════════════════════════════════════════════════════
# 5. REPORTE
# ═══════════════════════════════════════════════════════════════
def report(det, seg, names):
    sr_names = [n for n in names if n not in ('HR', 'LR (vecino)', 'Bicúbica')]
    print("\n" + "═" * 76)
    print("EVALUACIÓN BASADA EN TAREAS — BUSI test (modelos entrenados SOLO con HR)")
    print("═" * 76)
    rows = [[n, f"{det[n]['mAP50']:.3f}", f"{det[n]['mAP50_95']:.3f}", f"{det[n]['recall']:.3f}",
             f"{np.nanmean(det[n]['hit50']):.1%}", ms(det[n]['best_iou']),
             f"{np.nanmean(det[n]['fp_normal']):.2f}"] for n in names]
    print_table("1. Detección (YOLOv8n)",
                ['Versión', 'mAP50', 'mAP50-95', 'Recall', 'Lesiones IoU≥0.5', 'IoU mejor caja',
                 'Detecciones / normal'], rows)
    rows = [[n, ms(seg[n]['dice']), ms(seg[n]['iou']),
             f"{np.nanmean(seg[n]['fp_area_normal']):.4f}"] for n in names]
    print_table("2. Segmentación (U-Net)", ['Versión', 'Dice', 'IoU', 'Área falsa en normales'], rows)

    print("\nComparaciones pareadas frente a la bicúbica:")
    summary = {}
    for n in sr_names:
        s = {}
        for task, key, data in (('det', 'best_iou', det), ('seg', 'dice', seg), ('seg', 'iou', seg)):
            a, b, h = data[n][key], data['Bicúbica'][key], data['HR'][key]
            s[f'{task}_{key}'] = {'delta': float(np.nanmean(a - b)), 'p': wil(a, b),
                                  'better_frac': float(np.nanmean((a > b)[np.isfinite(a - b)])),
                                  'gap_recovered': gap(np.nanmean(a), np.nanmean(b), np.nanmean(h))}
            print(f"  {n:12s} {task}-{key:8s}: Δ {s[f'{task}_{key}']['delta']:+.4f} "
                  f"(mejor en {s[f'{task}_{key}']['better_frac']:.0%}, p={s[f'{task}_{key}']['p']:.1e}) "
                  f"· brecha recuperada {s[f'{task}_{key}']['gap_recovered']:.0%}")
        s['det_mAP50_gap'] = gap(det[n]['mAP50'], det['Bicúbica']['mAP50'], det['HR']['mAP50'])
        print(f"  {n:12s} mAP50: brecha recuperada {s['det_mAP50_gap']:.0%}")
        summary[n] = s
    return summary


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_json(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def main():
    t0 = time.time()
    D = load_data()
    print("\nReconstruyendo test:")
    V = build_versions(D['test'])
    names = list(V.keys())

    print("\n[1] Detección — entrenando YOLOv8n con HR nativa de train")
    det = eval_detection(train_detector(D), V, D['test'])
    print("\n[2] Segmentación — entrenando U-Net con HR nativa de train")
    seg = eval_segmentation(train_unet(D), V, D['test'])

    summary = report(det, seg, names)
    with open(os.path.join(OUT_DIR, 'tareas_busi.json'), 'w') as fh:
        json.dump(_json({'versions': names, 'det': det, 'seg': seg, 'summary': summary,
                         'config': {'det_imgsz': DET_IMGSZ, 'seg_size': SEG_SIZE,
                                    'sr_models': SR_MODELS}}), fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {OUT_DIR}/")


if __name__ == '__main__':
    main()
