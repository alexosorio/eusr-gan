# ═══════════════════════════════════════════════════════════════
# camus_eusr.py — EUSR-GAN v6 en CAMUS con el protocolo de Ahmad et al.
#
# Objetivo: comparación DIRECTA con Ahmad et al. (2022), Sci. Rep.
# 12:9533, que reportan en CAMUS (×4, HR 1024×512 → LR 256×128,
# 400 train / 100 test):
#     Bicúbica 33.21 dB / 0.89 · SRGAN 35.46 / 0.93 · Ahmad 37.11 / 0.95
#
# ARCHIVO AUTÓNOMO: incluye la arquitectura, las pérdidas y el
# entrenamiento en dos etapas de EUSR-GAN v6 (copiados sin cambios de
# eusr_gan_v6_yolo.py). No necesita ningún otro .py.
# Diferencias con BUSI: no hay lesiones → sin YOLO; los parches de
# entrenamiento se toman dentro del SECTOR del ultrasonido.
#
# Evaluación en test sobre la IMAGEN COMPLETA:
#   · PSNR/SSIM global (comparable con el artículo, incluye fondo negro)
#   · PSNR/SSIM solo dentro del sector (sin el fondo que infla el PSNR)
#
# Requisitos:  pip install -c constraints.txt nibabel
# Uso:  %run camus_eusr.py  (celda)  o  python camus_eusr.py  (terminal)
# Cita obligatoria: Leclerc et al., IEEE TMI 38(9):2198–2210, 2019.
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

# ── Configuración ─────────────────────────────────────────────
CAMUS_DIR = 'camus/database_nifti'
OUT_DIR   = 'resultados_camus'
SEED      = 42

VIEWS  = ['4CH']          # ['2CH', '4CH'] para usar ambas vistas
PHASES = ['ED']           # ['ED', 'ES'] para usar ambas fases
# Con 4CH+ED hay 1 imagen por paciente = 500, como en Ahmad et al.

HR_MODE = 'ahmad'         # 'ahmad': reescalar a 1024×512 (protocolo del artículo)
                          # 'native': resolución original (sin reescalar)
HR_H, HR_W = 1024, 512

N_TEST, N_VAL = 100, 40   # por PACIENTE; train = resto (360)
PATCHES_PER_IMG = 10      # parches 128×128 por imagen de train
VAL_PATCHES     = 4       # parches fijos por imagen de val
MIN_SECTOR      = 0.5     # fracción mínima del parche dentro del sector
FREE_FRAC       = 0.25    # fracción de parches sin restricción (borde y fondo
                          # negro: el modelo debe aprender a dejar el fondo en 0)

EPOCHS_S1  = 50
EPOCHS_S2  = 30
RUN_STAGE2 = True
TILE_STRIDE = 16          # inferencia por mosaicos LR 32×32 con solape

# Valores publicados por Ahmad et al. (Tabla de CAMUS)
AHMAD_REPORTED = {'Bicúbica (Ahmad)': (33.21, 1.5, 0.89, 0.20),
                  'SRGAN (Ahmad)':    (35.46, 2.1, 0.93, 0.20),
                  'Ahmad et al.':     (37.11, 1.7, 0.95, 0.23)}

# ── Arquitectura y entrenamiento (idénticos a EUSR-GAN v6 en BUSI) ──
SCALE      = 4
HR_PATCH   = 128
LR_PATCH   = HR_PATCH // SCALE     # 32
BATCH_SIZE = 4
LR_S1      = 5e-5
EARLY_STOP_S1 = 16
GN_GROUPS  = 8
NUM_RRDB   = 4
GROWTH_CH  = 32
CH         = 64

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
W_S1 = os.path.join(OUT_DIR, 'g_camus_stage1_best.weights.h5')
W_S2 = os.path.join(OUT_DIR, 'g_camus_stage2_best.weights.h5')
P_HR = HR_PATCH


# ═══════════════════════════════════════════════════════════════
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
# C1. CAMUS — INVENTARIO Y CARGA
# ═══════════════════════════════════════════════════════════════
def read_cfg(fp):
    info = {}
    if os.path.exists(fp):
        for line in open(fp):
            if ':' in line:
                k, v = line.split(':', 1)
                info[k.strip()] = v.strip()
    return info


def list_camus(root=CAMUS_DIR):
    items = []
    for pdir in sorted(glob.glob(os.path.join(root, 'patient*'))):
        pid = os.path.basename(pdir)
        for view in VIEWS:
            info = read_cfg(os.path.join(pdir, f'Info_{view}.cfg'))
            for phase in PHASES:
                fp = os.path.join(pdir, f'{pid}_{view}_{phase}.nii.gz')
                if os.path.exists(fp):
                    items.append({'img': fp, 'patient': pid, 'view': view,
                                  'phase': phase,
                                  'quality': info.get('ImageQuality', 'NA')})
    return items


def load_nifti(fp):
    """
    nibabel devuelve (x, y) = (columnas, filas); se transpone a (filas,
    columnas). Revisa camus_preview.png: el vértice del sector debe
    quedar arriba.
    """
    import nibabel as nib
    a = np.asarray(nib.load(fp).dataobj, dtype=np.float32).squeeze().T
    return a / 255.0 if a.max() > 1.5 else a


def to_hr(img):
    if HR_MODE == 'ahmad':
        im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
        return np.asarray(im.resize((HR_W, HR_H), Image.BICUBIC),
                          dtype=np.float32) / 255.0
    H, W = img.shape
    return img[:H - H % SCALE, :W - W % SCALE].copy()


def sector_mask(img, thr=0.01):
    """Región del abanico: no-negro, cerrada y sin huecos (componente mayor)."""
    m = ndimage.binary_closing(img > thr, structure=np.ones((7, 7)), iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return m


def split_patients(items, seed=SEED):
    pats = sorted({it['patient'] for it in items})
    rng = np.random.default_rng(seed)
    pats = [pats[i] for i in rng.permutation(len(pats))]
    te, va = set(pats[:N_TEST]), set(pats[N_TEST:N_TEST + N_VAL])
    tr = [it for it in items if it['patient'] not in te | va]
    return (tr, [it for it in items if it['patient'] in va],
            [it for it in items if it['patient'] in te])


# ═══════════════════════════════════════════════════════════════
# C2. CAMUS — PARCHES (dentro del sector)
# ═══════════════════════════════════════════════════════════════
def sample_patches(hr, mask, n, rng, do_aug):
    H, W = hr.shape
    integ = mask.astype(np.float32).cumsum(0).cumsum(1)
    out = []
    for _ in range(n):
        libre = rng.random() < FREE_FRAC
        for _try in range(1 if libre else 50):
            y = int(rng.integers(0, H - P_HR + 1))
            x = int(rng.integers(0, W - P_HR + 1))
            s = integ[y + P_HR - 1, x + P_HR - 1]
            s -= integ[y - 1, x + P_HR - 1] if y > 0 else 0
            s -= integ[y + P_HR - 1, x - 1] if x > 0 else 0
            s += integ[y - 1, x - 1] if (y > 0 and x > 0) else 0
            if s / (P_HR * P_HR) >= MIN_SECTOR:
                break
        p = hr[y:y + P_HR, x:x + P_HR]
        out.append(augment(p, rng) if do_aug else p.copy())
    return out


def build_patch_set(hrs, masks, n_per, seed, do_aug):
    rng = np.random.default_rng(seed)
    hr_p = []
    for hr, m in zip(hrs, masks):
        hr_p += sample_patches(hr, m, n_per, rng, do_aug)
    hr_p = np.stack(hr_p)[..., None].astype(np.float32)
    lr_p = np.stack([degrade(p[..., 0]) for p in hr_p])[..., None]
    return lr_p, hr_p


# ═══════════════════════════════════════════════════════════════
# C3. CAMUS — INFERENCIA SOBRE IMAGEN COMPLETA (mosaicos, admite rectangulares)
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# C4. CAMUS — MÉTRICAS: imagen completa y solo sector
# ═══════════════════════════════════════════════════════════════
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


def evaluate_test(G, test):
    rows = []
    for it, hr in zip(test['items'], test['hr']):
        lr  = degrade(hr)
        bic = bicubic_full(lr, hr.shape)
        sr  = sr_tiles(G, lr)
        m   = sector_mask(hr)
        r = {'patient': it['patient'], 'view': it['view'], 'phase': it['phase'],
             'quality': it['quality']}
        for name, img in (('bic', bic), ('sr', sr)):
            r[f'{name}_psnr']        = psnr(hr, img)
            r[f'{name}_ssim']        = ssim_full(hr, img)
            r[f'{name}_psnr_sector'] = psnr(hr, img, m)
            r[f'{name}_ssim_sector'] = float(ssim_map(hr, img)[m].mean())
        rows.append(r)
    return rows


def summarize(rows):
    def col(k):
        return np.array([r[k] for r in rows], float)

    out = {}
    for metric in ('psnr', 'ssim', 'psnr_sector', 'ssim_sector'):
        b, s = col(f'bic_{metric}'), col(f'sr_{metric}')
        try:
            p = float(stats.wilcoxon(s, b).pvalue)
        except ValueError:
            p = float('nan')
        out[metric] = {'bicubica': (float(b.mean()), float(b.std())),
                       'eusr':     (float(s.mean()), float(s.std())),
                       'delta':    float((s - b).mean()), 'p_wilcoxon': p}
    return out


def print_report(summ, rows, n_train):
    print("\n" + "═" * 72)
    print(f"CAMUS — TEST ({len(rows)} imágenes, ×{SCALE}, modo HR = {HR_MODE})")
    print("═" * 72)
    print("Imagen completa (comparable con Ahmad et al.):")
    print(f"  {'Método':22s} {'PSNR (dB)':>16s} {'SSIM':>16s}")
    for k, (p, ps, s, ss) in AHMAD_REPORTED.items():
        print(f"  {k:22s} {p:8.2f} ± {ps:<5.2f} {s:8.3f} ± {ss:<5.2f}  (reportado)")
    for key, name in (('bicubica', 'Bicúbica (nuestra)'), ('eusr', 'EUSR-GAN v6')):
        p, s = summ['psnr'][key], summ['ssim'][key]
        print(f"  {name:22s} {p[0]:8.2f} ± {p[1]:<5.2f} {s[0]:8.3f} ± {s[1]:<5.3f}")
    print(f"  Δ EUSR − bicúbica: {summ['psnr']['delta']:+.2f} dB "
          f"(p={summ['psnr']['p_wilcoxon']:.1e}) | SSIM {summ['ssim']['delta']:+.4f} "
          f"(p={summ['ssim']['p_wilcoxon']:.1e})")
    print("\nSolo dentro del sector (sin fondo negro):")
    for key, name in (('bicubica', 'Bicúbica'), ('eusr', 'EUSR-GAN v6')):
        p, s = summ['psnr_sector'][key], summ['ssim_sector'][key]
        print(f"  {name:22s} {p[0]:8.2f} ± {p[1]:<5.2f} {s[0]:8.3f} ± {s[1]:<5.3f}")
    print(f"  Δ EUSR − bicúbica: {summ['psnr_sector']['delta']:+.2f} dB | "
          f"SSIM {summ['ssim_sector']['delta']:+.4f}")
    print("\nPor calidad de imagen (PSNR completa, EUSR − bicúbica):")
    for q in sorted({r['quality'] for r in rows}):
        g = [r for r in rows if r['quality'] == q]
        d = np.mean([r['sr_psnr'] - r['bic_psnr'] for r in g])
        print(f"  {q:8s} n={len(g):3d}  EUSR {np.mean([r['sr_psnr'] for r in g]):.2f} dB  Δ {d:+.2f} dB")
    print(f"\nTrain: {n_train} imágenes. Nota: Ahmad et al. no usan conjunto de validación;"
          " aquí se separan 40 pacientes de train para seleccionar el modelo.")


# ═══════════════════════════════════════════════════════════════
# C5. CAMUS — FIGURAS
# ═══════════════════════════════════════════════════════════════
def save_preview(hrs, masks, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(3, len(hrs))
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 5))
    for i, a in enumerate(np.atleast_1d(ax)):
        a.imshow(hrs[i], cmap='gray', vmin=0, vmax=1)
        a.contour(masks[i], levels=[0.5], colors='lime', linewidths=0.8)
        a.set_title(f'{hrs[i].shape[0]}×{hrs[i].shape[1]}')
        a.axis('off')
    fig.suptitle('Revisión: vértice del sector arriba · verde = sector detectado')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_examples(G, test, path, n=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(n, len(test['hr']))
    fig, ax = plt.subplots(n, 4, figsize=(13, 3.6 * n), squeeze=False)
    for r in range(n):
        hr = test['hr'][r]
        lr = degrade(hr)
        imgs = [bicubic_full(lr, hr.shape), sr_tiles(G, lr), hr]
        H, W = hr.shape
        y0, x0, s = H // 3, W // 3, min(H, W) // 3          # recorte central
        ax[r, 0].imshow(lr, cmap='gray', vmin=0, vmax=1)
        ax[r, 0].set_title(f'LR {lr.shape[0]}×{lr.shape[1]}')
        for c, (img, t) in enumerate(zip(imgs, ['Bicúbica', 'EUSR-GAN v6', 'HR']), 1):
            ax[r, c].imshow(img[y0:y0 + s, x0:x0 + s], cmap='gray', vmin=0, vmax=1)
            ax[r, c].set_title(f'{t} (recorte)')
        for a in ax[r]:
            a.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# C6. CAMUS — MAIN
# ═══════════════════════════════════════════════════════════════
def load_set(items):
    raw = [load_nifti(it['img']) for it in items]
    hrs = [to_hr(a) for a in raw]
    return {'items': items, 'raw_shapes': [a.shape for a in raw], 'hr': hrs,
            'mask': [sector_mask(h) for h in hrs]}


def main():
    t0 = time.time()
    items = list_camus()
    if not items:
        raise SystemExit(f"No se encontraron imágenes en {CAMUS_DIR}")
    tr_i, va_i, te_i = split_patients(items)
    print(f"CAMUS: {len(items)} imágenes | vistas {VIEWS} | fases {PHASES}")
    print(f"Pacientes → train {len({i['patient'] for i in tr_i})}  "
          f"val {len({i['patient'] for i in va_i})}  test {len({i['patient'] for i in te_i})}")

    sets = {k: load_set(v) for k, v in (('train', tr_i), ('val', va_i), ('test', te_i))}
    shapes = np.array([s for k in sets for s in sets[k]['raw_shapes']])
    print(f"Tamaño original (filas×columnas): mín {shapes.min(0)}  "
          f"mediana {np.median(shapes, 0).astype(int)}  máx {shapes.max(0)}")
    if HR_MODE == 'ahmad':
        up = np.mean((shapes[:, 0] < HR_H) | (shapes[:, 1] < HR_W))
        print(f"Imágenes que se AMPLÍAN para llegar a {HR_H}×{HR_W}: {up:.0%}"
              + ("  ⚠ su HR es interpolado, no real" if up > 0 else ""))
    save_preview(sets['train']['hr'], sets['train']['mask'],
                 os.path.join(OUT_DIR, 'camus_preview.png'))

    x_tr, y_tr = build_patch_set(sets['train']['hr'], sets['train']['mask'],
                                 PATCHES_PER_IMG, SEED, do_aug=True)
    x_va, y_va = build_patch_set(sets['val']['hr'], sets['val']['mask'],
                                 VAL_PATCHES, SEED + 1, do_aug=False)
    print(f"Parches → train {len(x_tr)}  val {len(x_va)}")

    G = build_generator()
    G, hist_s1, best_s1 = train_stage1(G, x_tr, y_tr, x_va, y_va, epochs=EPOCHS_S1)
    stage2 = False
    if RUN_STAGE2:
        D = build_discriminator()
        G, hist_s2, stage2 = train_stage2(G, D, x_tr, y_tr, x_va, y_va,
                                             best_s1, epochs=EPOCHS_S2)

    print("\nEvaluando test sobre imágenes completas…")
    rows = evaluate_test(G, sets['test'])
    summ = summarize(rows)
    print_report(summ, rows, len(tr_i))
    save_examples(G, sets['test'], os.path.join(OUT_DIR, 'camus_ejemplos.png'))

    with open(os.path.join(OUT_DIR, 'resultados_camus.json'), 'w') as fh:
        json.dump({'config': {'views': VIEWS, 'phases': PHASES, 'hr_mode': HR_MODE,
                              'hr_size': [HR_H, HR_W], 'scale': SCALE,
                              'n_test': len(te_i), 'n_val': len(va_i), 'n_train': len(tr_i),
                              'stage2_used': bool(stage2)},
                   'ahmad_reported': AHMAD_REPORTED, 'resumen': summ,
                   'por_imagen': rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {OUT_DIR}/")


if __name__ == '__main__':
    main()
