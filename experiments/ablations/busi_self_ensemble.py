# ═══════════════════════════════════════════════════════════════
# busi_native.py — EUSR-GAN v6 en BUSI a RESOLUCIÓN NATIVA
#
# Mismo protocolo que la corrida nativa de CAMUS (camus_eusr.py con
# HR_MODE='native'): la HR es la imagen original de BUSI (~500 px), sin
# reducirla a 256, y la LR es la HR reducida ×4 con bicúbica. Así BUSI y
# CAMUS se comparan con el mismo modelo y el mismo protocolo.
#
# Opciones (corridas propuestas):
#   B1  INIT='scratch'  PATCH_MODE='yolo'    resultado base
#   B2  INIT='camus'    PATCH_MODE='yolo'    transferencia desde CAMUS
#   B3  (mejor INIT)    PATCH_MODE='tissue'  ablación: sin guía de YOLO
#
# · Split IDÉNTICO a v6 (semilla 42, mismos duplicados eliminados):
#   el detector YOLO de v6 se entrenó con ese split → sin filtración.
# · YOLO detecta en la versión 256 (como fue entrenado) y las cajas se
#   reescalan a la resolución nativa.
# · Validación (para elegir el modelo) igual en todas las corridas:
#   parches al azar dentro del tejido, sin YOLO.
# · Test sobre la imagen COMPLETA (mosaicos), en tres zonas:
#   imagen completa · solo tejido (sin bordes negros) · solo lesión.
#
# ARCHIVO AUTÓNOMO: incluye arquitectura, pérdidas y entrenamiento de
# EUSR-GAN v6 sin cambios. Requiere ultralytics solo si PATCH_MODE='yolo'.
# Uso:  %run busi_native.py  (celda)  o  python busi_native.py  (terminal)
# ═══════════════════════════════════════════════════════════════

import os
import glob
import json
import time

import numpy as np
from PIL import Image
from scipy import ndimage, stats
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, GroupNormalization, ReLU, LeakyReLU,
    Add, Multiply, Concatenate, Input, Dense,
    GlobalAveragePooling2D, Lambda, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam

# ── Configuración de la corrida ───────────────────────────────
INIT = 'scratch'
PATCH_MODE = 'yolo'
TRAIN_SEED  = 42          # cambiar (7, 123…) para repetir con otra semilla
USE_FFT     = True        # False = ablación sin la pérdida FFT del speckle
CAMUS_WEIGHTS = 'resultados_camus_native/g_camus_stage1_best.weights.h5'
INIT_WEIGHTS = 'resultados_busi_native_scratch_yolo/g_busi_stage1_best.weights.h5'

# ── Mejoras opcionales (probar UNA por corrida) ───────────────
RESIDUAL_BICUBIC = False  # arquitectura: la red predice el residuo sobre la bicúbica
NUM_RRDB         = 4      # arquitectura: bloques RRDB del tronco (v6 = 4)
SELF_ENSEMBLE = True
COMPARE_WITH = ['resultados_busi_native_scratch_yolo/resultados_busi.json']

# ── Datos (NO cambiar: deben coincidir con v6 y con YOLO) ─────
DATASET     = 'busi/Dataset_BUSI_with_GT'
SEED        = 42          # semilla del split de v6
DEDUP       = True
DEDUP_THUMB = 64
DEDUP_MAD   = 0.008
SRC_SIZE    = 256         # tamaño con que se entrenó YOLO
RG_THRESHOLD = 0.15
YOLO_BEST   = 'yolo_busi/lesion/weights/best.pt'
YOLO_CONF   = 0.25

# ── Parches ───────────────────────────────────────────────────
PATCHES_PER_IMG = 3       # como v6 (train balanceado por clase)
VAL_PATCHES     = 4
JITTER_FRAC     = 24 / 256   # desplazamiento del centro, relativo al tamaño
TILE_STRIDE     = 16

# ── Entrenamiento (como CAMUS nativo) ─────────────────────────
EPOCHS_S1 = 0
EPOCHS_S2  = 30
RUN_STAGE2 = False
FFT_WEIGHT = 0.05 if USE_FFT else 0.0

_tag_init = INIT if INIT_WEIGHTS is None else 'pesos'
OUT_DIR = (f'resultados_busi_native_{_tag_init}_{PATCH_MODE}'
           + ('' if USE_FFT else '_nofft')
           + ('_res' if RESIDUAL_BICUBIC else '')
           + ('' if NUM_RRDB == 4 else f'_rrdb{NUM_RRDB}')
           + ('_se8' if SELF_ENSEMBLE else '')
           + ('' if EPOCHS_S1 > 0 else ('_eval' if INIT_WEIGHTS else '_zeroshot'))
           + ('' if TRAIN_SEED == 42 else f'_s{TRAIN_SEED}'))

# ── Arquitectura (idéntica a EUSR-GAN v6) ─────────────────────
SCALE      = 4
HR_PATCH   = 128
LR_PATCH   = HR_PATCH // SCALE
BATCH_SIZE = 4
LR_S1      = 5e-5
EARLY_STOP_S1 = 16
GN_GROUPS  = 8
GROWTH_CH  = 32
CH         = 64

tf.keras.utils.set_random_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
os.makedirs(OUT_DIR, exist_ok=True)
W_S1 = os.path.join(OUT_DIR, 'g_busi_stage1_best.weights.h5')
W_S2 = os.path.join(OUT_DIR, 'g_busi_stage2_best.weights.h5')
P_HR = HR_PATCH


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


def load_gray(fp: str, size: int = SRC_SIZE) -> np.ndarray:
    im = Image.open(fp).convert('L').resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


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
    return ndimage.binary_fill_holes(m)# ═══════════════════════════════════════════════════════════════
# A. DEGRADACIÓN Y AUMENTO — idénticos a v6
# ═══════════════════════════════════════════════════════════════
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
    if RESIDUAL_BICUBIC:
        # la red aprende solo el detalle que le falta a la bicúbica
        res  = Conv2D(1, 3, padding='same', name='hr_residual',
                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 1e-3))(p3)
        base = Lambda(lambda t: tf.image.resize(t, (lr_size * SCALE, lr_size * SCALE),
                                                method='bicubic'), name='bicubic_base')(inp)
        out  = Lambda(lambda t: tf.clip_by_value(t, 0.0, 1.0), name='hr_output')(Add()([base, res]))
    else:
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
                            lambda_freq=FFT_WEIGHT, lambda_perc=0.05,
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



def _positions(n, t, stride):
    p = list(range(0, n - t + 1, stride))
    if p[-1] != n - t:
        p.append(n - t)
    return p


def sr_tiles(G, lr, stride=TILE_STRIDE):
    t, T = LR_PATCH, LR_PATCH * SCALE
    h, w = lr.shape
    w1  = np.hanning(T + 2)[1:-1]
    win = np.outer(w1, w1).astype(np.float32)
    coords = [(y, x) for y in _positions(h, t, stride) for x in _positions(w, t, stride)]
    tiles = np.stack([lr[y:y + t, x:x + t] for y, x in coords])[..., None]
    pred = predict_sr(G, tiles, batch=64)[..., 0]
    out = np.zeros((h * SCALE, w * SCALE), np.float32)
    acc = np.zeros_like(out)
    for (y, x), p in zip(coords, pred):
        Y, X = y * SCALE, x * SCALE
        out[Y:Y + T, X:X + T] += p * win
        acc[Y:Y + T, X:X + T] += win
    return np.clip(out / acc, 0.0, 1.0)


def bicubic_full(lr, shape):
    return np.clip(tf.image.resize(lr[None, ..., None], shape,
                                   method='bicubic').numpy()[0, ..., 0], 0, 1)


def psnr(a, b, m=None):
    d = (a - b) ** 2
    mse = d[m].mean() if m is not None else d.mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def ssim_full(a, b):
    return float(tf.image.ssim(a[None, ..., None], b[None, ..., None], max_val=1.0)[0])


def ssim_map(x, y, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = lambda a: ndimage.gaussian_filter(a, sigma, truncate=3.5)   # noqa: E731
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx ** 2, g(y * y) - my ** 2, g(x * y) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))

# ═══════════════════════════════════════════════════════════════
# B1. CARGA NATIVA
# ═══════════════════════════════════════════════════════════════
def load_native(it):
    """Imagen y máscara a resolución original, recortadas a múltiplo de 4."""
    im = Image.open(it['img']).convert('L')
    W0, H0 = im.size
    img = np.asarray(im, dtype=np.float32) / 255.0
    m = np.zeros((H0, W0), dtype=bool)
    for mp in it['masks']:
        mm = Image.open(mp).convert('L')
        if mm.size != (W0, H0):
            mm = mm.resize((W0, H0), Image.NEAREST)
        m |= np.asarray(mm) > 127
    H, W = H0 - H0 % SCALE, W0 - W0 % SCALE
    img, m = img[:H, :W], m[:H, :W]
    # guarda: imágenes menores que un parche → relleno negro abajo/derecha
    ph, pw = max(0, P_HR - H), max(0, P_HR - W)
    valid = np.ones((H, W), dtype=bool)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)))
        m = np.pad(m, ((0, ph), (0, pw)))
        valid = np.pad(valid, ((0, ph), (0, pw)))
    return img, m, valid, (H0, W0)


def tissue_region(img, valid, thr=0.02):
    """Zona de imagen útil: no negra, cerrada y sin huecos (componente mayor)."""
    m = ndimage.binary_closing((img > thr) & valid, structure=np.ones((7, 7)), iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return m & valid


def load_set(items):
    out = {'items': items, 'hr': [], 'mask': [], 'valid': [], 'tissue': [],
           'rg': [], 'orig': []}
    for it in items:
        img, m, valid, orig = load_native(it)
        out['hr'].append(img); out['mask'].append(m); out['valid'].append(valid)
        out['tissue'].append(tissue_region(img, valid))
        out['rg'].append(tissue_mask(img) & valid)     # region growing de v6
        out['orig'].append(orig)
    return out


# ═══════════════════════════════════════════════════════════════
# B2. YOLO (detecta en 256 → cajas a resolución nativa)
# ═══════════════════════════════════════════════════════════════
def detect_native(items, origs):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠ ultralytics no está instalado → sin detecciones (fallback a tejido)")
        return {}
    if not os.path.exists(YOLO_BEST):
        print(f"⚠ No existe {YOLO_BEST} → sin detecciones (fallback a tejido)")
        return {}
    model, dets = YOLO(YOLO_BEST), {}
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        arrs = [np.repeat((load_gray(it['img']) * 255).astype(np.uint8)[..., None], 3, -1)
                for it in chunk]
        res = model.predict(source=arrs, imgsz=SRC_SIZE, conf=YOLO_CONF,
                            device='cpu', verbose=False)
        for it, r, (H0, W0) in zip(chunk, res, origs[i:i + 16]):
            sx, sy = W0 / SRC_SIZE, H0 / SRC_SIZE
            b = r.boxes
            if b is None or len(b) == 0:
                dets[it['id']] = []
                continue
            xyxy, cf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
            dets[it['id']] = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy, float(c))
                              for (x1, y1, x2, y2), c in sorted(zip(xyxy, cf), key=lambda t: -t[1])]
    return dets


# ═══════════════════════════════════════════════════════════════
# B3. PARCHES
# ═══════════════════════════════════════════════════════════════
def _rand_point(mask, rng, H, W):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return W / 2, H / 2, 'centro'
    k = int(rng.integers(len(ys)))
    return float(xs[k]), float(ys[k]), 'tejido'


def pick_center(boxes, rg, H, W, rng):
    if PATCH_MODE == 'yolo' and boxes:
        conf = np.array([b[4] for b in boxes])
        x1, y1, x2, y2, _ = boxes[int(rng.choice(len(boxes), p=conf / conf.sum()))]
        j = JITTER_FRAC * min(H, W)
        return ((x1 + x2) / 2 + rng.uniform(-j, j),
                (y1 + y2) / 2 + rng.uniform(-j, j), 'yolo')
    return _rand_point(rg, rng, H, W)


def extract(img, cx, cy):
    H, W = img.shape
    top = int(np.clip(round(cy - P_HR / 2), 0, H - P_HR))
    left = int(np.clip(round(cx - P_HR / 2), 0, W - P_HR))
    return img[top:top + P_HR, left:left + P_HR].copy()


def build_train(S, dets, rng):
    """Oversample por clase (solo train) + PATCHES_PER_IMG parches aumentados."""
    cls = np.array([it['clase'] for it in S['items']])
    n_max = max((cls == c).sum() for c in set(cls))
    idx = []
    for c in sorted(set(cls)):
        ii = np.where(cls == c)[0]
        idx += list(ii) + list(rng.choice(ii, n_max - len(ii), replace=True))
    patches, origen = [], {'yolo': 0, 'tejido': 0, 'centro': 0}
    for i in idx:
        img, it = S['hr'][i], S['items'][i]
        H, W = img.shape
        for _ in range(PATCHES_PER_IMG):
            cx, cy, o = pick_center(dets.get(it['id'], []), S['rg'][i], H, W, rng)
            origen[o] += 1
            patches.append(augment(extract(img, cx, cy), rng))
    print(f"  train: {len(patches)} parches ({len(idx)} imágenes tras balancear) · origen {origen}")
    hr = np.stack(patches)[..., None].astype(np.float32)
    return np.stack([degrade(p[..., 0]) for p in hr])[..., None], hr


def build_val(S, rng):
    """Igual para todas las corridas: parches al azar en el tejido, sin YOLO."""
    patches = []
    for img, tis in zip(S['hr'], S['tissue']):
        H, W = img.shape
        for _ in range(VAL_PATCHES):
            cx, cy, _ = _rand_point(tis, rng, H, W)
            patches.append(extract(img, cx, cy))
    hr = np.stack(patches)[..., None].astype(np.float32)
    return np.stack([degrade(p[..., 0]) for p in hr])[..., None], hr


# ═══════════════════════════════════════════════════════════════
# B4. EVALUACIÓN EN TEST (imagen completa)
# ═══════════════════════════════════════════════════════════════
def lesion_box(m, pad=8, min_size=32):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    H, W = m.shape

    def fix(a, b, n):
        a, b = max(0, a - pad), min(n, b + pad)
        if b - a < min_size:
            c = (a + b) // 2
            a, b = max(0, c - min_size // 2), min(n, max(0, c - min_size // 2) + min_size)
        return a, b
    y0, y1 = fix(ys.min(), ys.max() + 1, H)
    x0, x1 = fix(xs.min(), xs.max() + 1, W)
    return y0, y1, x0, x1


def sr_infer(G, lr):
    """Inferencia en mosaicos; con SELF_ENSEMBLE promedia las 8 orientaciones."""
    if not SELF_ENSEMBLE:
        return sr_tiles(G, lr)
    acc = None
    for k in range(4):
        for flip in (False, True):
            x = np.rot90(lr, k)
            x = x[:, ::-1] if flip else x
            y = sr_tiles(G, np.ascontiguousarray(x))
            y = y[:, ::-1] if flip else y
            y = np.rot90(y, -k)
            acc = y if acc is None else acc + y
    return acc / 8.0


def evaluate_test(G, S):
    rows = []
    for i, it in enumerate(S['items']):
        hr, valid = S['hr'][i], S['valid'][i]
        lr = degrade(hr)
        imgs = {'bic': bicubic_full(lr, hr.shape), 'sr': sr_infer(G, lr)}
        H0, W0 = np.nonzero(valid)[0].max() + 1, np.nonzero(valid)[1].max() + 1
        r = {'id': it['id'], 'clase': it['clase'], 'size': list(S['orig'][i])}
        tis, box = S['tissue'][i], lesion_box(S['mask'][i])
        for k, img in imgs.items():
            h, s = hr[:H0, :W0], img[:H0, :W0]
            r[f'{k}_psnr'] = psnr(h, s)
            r[f'{k}_ssim'] = ssim_full(h, s)
            r[f'{k}_psnr_tejido'] = psnr(hr, img, tis)
            r[f'{k}_ssim_tejido'] = float(ssim_map(hr, img)[tis].mean())
            if box is not None:
                y0, y1, x0, x1 = box
                r[f'{k}_psnr_lesion'] = psnr(hr[y0:y1, x0:x1], img[y0:y1, x0:x1])
                r[f'{k}_ssim_lesion'] = ssim_full(hr[y0:y1, x0:x1], img[y0:y1, x0:x1])
        rows.append(r)
    return rows


def _col(rows, k):
    return np.array([r[k] for r in rows if k in r], float)


def _ms(x):
    return f"{x.mean():6.2f} ± {x.std():.2f}" if x.mean() > 1.5 else f"{x.mean():6.4f} ± {x.std():.3f}"


def _wil(a, b):
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float('nan')


def report(rows):
    summ = {}
    print("\n" + "═" * 76)
    print(f"BUSI NATIVO — TEST ({len(rows)} imágenes, ×{SCALE}) · INIT={INIT} · "
          f"PARCHES={PATCH_MODE} · FFT={'sí' if USE_FFT else 'no'}"
          + (" · RESIDUAL" if RESIDUAL_BICUBIC else "")
          + (f" · RRDB={NUM_RRDB}" if NUM_RRDB != 4 else "")
          + (" · AUTOENSAMBLE×8" if SELF_ENSEMBLE else ""))
    print("═" * 76)
    for zona, suf in (('Imagen completa', ''), ('Solo tejido', '_tejido'), ('Solo lesión', '_lesion')):
        sub = [r for r in rows if f'sr_psnr{suf}' in r]
        print(f"\n{zona} (n={len(sub)}):")
        for k, nom in (('bic', 'Bicúbica'), ('sr', 'EUSR-GAN v6')):
            print(f"  {nom:14s} PSNR {_ms(_col(sub, f'{k}_psnr{suf}'))} dB   "
                  f"SSIM {_ms(_col(sub, f'{k}_ssim{suf}'))}")
        z = {}
        for met in ('psnr', 'ssim'):
            a, b = _col(sub, f'sr_{met}{suf}'), _col(sub, f'bic_{met}{suf}')
            z[met] = {'bic': [float(b.mean()), float(b.std())],
                      'eusr': [float(a.mean()), float(a.std())],
                      'delta': float((a - b).mean()), 'mejor_frac': float(np.mean(a > b)),
                      'p': _wil(a, b)}
        print(f"  Δ EUSR − bicúbica: PSNR {z['psnr']['delta']:+.2f} dB "
              f"(mejor en {z['psnr']['mejor_frac']:.0%}, p={z['psnr']['p']:.1e}) · "
              f"SSIM {z['ssim']['delta']:+.4f} (p={z['ssim']['p']:.1e})")
        summ[zona] = z
    print("\nPor clase (imagen completa):")
    for c in sorted({r['clase'] for r in rows}):
        g = [r for r in rows if r['clase'] == c]
        a, b = _col(g, 'sr_psnr'), _col(g, 'bic_psnr')
        print(f"  {c:10s} n={len(g):3d}  EUSR {a.mean():.2f} dB  bicúbica {b.mean():.2f} dB  "
              f"Δ {(a - b).mean():+.2f} dB")
    for path in COMPARE_WITH:
        if not os.path.exists(path):
            print(f"\n(No existe {path}; se omite la comparación.)")
            continue
        other = {r['id']: r for r in json.load(open(path))['por_imagen']}
        com = [r for r in rows if r['id'] in other]
        print(f"\nComparación pareada con {path} (n={len(com)}):")
        for suf in ('', '_tejido', '_lesion'):
            cc = [r for r in com if f'sr_psnr{suf}' in r and f'sr_psnr{suf}' in other[r['id']]]
            a = np.array([r[f'sr_psnr{suf}'] for r in cc])
            b = np.array([other[r['id']][f'sr_psnr{suf}'] for r in cc])
            print(f"  PSNR{suf or ' completa':9s}: esta − otra = {(a - b).mean():+.3f} dB "
                  f"(mejor en {np.mean(a > b):.0%}, p={_wil(a, b):.1e})")
    return summ


def save_examples(G, S, path, n=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idx = [i for i, m in enumerate(S['mask']) if m.any()][:n]
    if not idx:
        return
    fig, ax = plt.subplots(len(idx), 4, figsize=(13, 3.4 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        hr = S['hr'][i]
        lr = degrade(hr)
        ys, xs = np.nonzero(S['mask'][i])
        cy, cx, s = int(ys.mean()), int(xs.mean()), 160
        y0 = int(np.clip(cy - s // 2, 0, max(0, hr.shape[0] - s)))
        x0 = int(np.clip(cx - s // 2, 0, max(0, hr.shape[1] - s)))
        crops = [np.kron(lr, np.ones((SCALE, SCALE))), bicubic_full(lr, hr.shape),
                 sr_tiles(G, lr), hr]
        for c, (img, t) in enumerate(zip(crops, ['LR (×4 vecino)', 'Bicúbica', 'EUSR-GAN v6', 'HR nativa'])):
            ax[r, c].imshow(img[y0:y0 + s, x0:x0 + s], cmap='gray', vmin=0, vmax=1)
            ax[r, c].set_title(t if r else f'{t}')
            ax[r, c].axis('off')
        ax[r, 0].text(2, 12, S['items'][i]['clase'], color='yellow', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# B5. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    items = list_busi(DATASET)
    print(f"BUSI: {len(items)} imágenes")
    if DEDUP:
        items = dedupe(items)
    print("Split (idéntico a v6):")
    tr_i, va_i, te_i = split_items(items)
    S = {k: load_set(v) for k, v in (('train', tr_i), ('val', va_i), ('test', te_i))}
    sizes = np.array([o for k in S for o in S[k]['orig']])
    print(f"Tamaño original (filas×columnas): mín {sizes.min(0)}  "
          f"mediana {np.median(sizes, 0).astype(int)}  máx {sizes.max(0)}")

    dets = detect_native(tr_i, S['train']['orig']) if PATCH_MODE == 'yolo' else {}
    rng = np.random.default_rng(TRAIN_SEED)
    print("Parches:")
    x_tr, y_tr = build_train(S['train'], dets, rng)
    x_va, y_va = build_val(S['val'], np.random.default_rng(SEED + 1))
    print(f"  val: {len(x_va)} parches (al azar en tejido, iguales en todas las corridas)")

    G = build_generator()
    if INIT == 'camus' and (RESIDUAL_BICUBIC or NUM_RRDB != 4):
        raise SystemExit("Los pesos de CAMUS son de la arquitectura v6 original: "
                         "usa INIT='scratch' con RESIDUAL_BICUBIC o NUM_RRDB != 4")
    if INIT_WEIGHTS:
        if not os.path.exists(INIT_WEIGHTS):
            raise SystemExit(f"No existe {INIT_WEIGHTS}")
        G.load_weights(INIT_WEIGHTS)
        print(f"Pesos cargados: {INIT_WEIGHTS}")
    elif INIT == 'camus':
        if not os.path.exists(CAMUS_WEIGHTS):
            raise SystemExit(f"No existe {CAMUS_WEIGHTS}")
        G.load_weights(CAMUS_WEIGHTS)
        p0, s0 = evaluate(G, x_va, y_va)
        print(f"Pesos de CAMUS cargados · val BUSI antes de ajustar: PSNR {p0:.2f} dB  SSIM {s0:.4f}")
    stage2 = False
    if EPOCHS_S1 > 0:
        G, hist_s1, best_s1 = train_stage1(G, x_tr, y_tr, x_va, y_va, epochs=EPOCHS_S1)
    else:
        print("EPOCHS_S1 = 0 → sin entrenamiento: se evalúan los pesos cargados tal cual")
    if RUN_STAGE2 and EPOCHS_S1 > 0:
        D = build_discriminator()
        G, hist_s2, stage2 = train_stage2(G, D, x_tr, y_tr, x_va, y_va, best_s1, epochs=EPOCHS_S2)

    print("\nEvaluando test sobre imágenes completas…")
    rows = evaluate_test(G, S['test'])
    summ = report(rows)
    save_examples(G, S['test'], os.path.join(OUT_DIR, 'busi_ejemplos.png'))
    with open(os.path.join(OUT_DIR, 'resultados_busi.json'), 'w') as fh:
        json.dump({'config': {'init': INIT, 'patch_mode': PATCH_MODE, 'train_seed': TRAIN_SEED,
                              'use_fft': USE_FFT, 'epochs_s1': EPOCHS_S1,
                              'residual_bicubic': RESIDUAL_BICUBIC, 'num_rrdb': NUM_RRDB,
                              'self_ensemble': SELF_ENSEMBLE, 'init_weights': INIT_WEIGHTS,
                              'split_seed': SEED, 'stage2_used': bool(stage2),
                              'n_train': len(tr_i), 'n_val': len(va_i), 'n_test': len(te_i)},
                   'resumen': summ, 'por_imagen': rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {OUT_DIR}/")


if __name__ == '__main__':
    main()
