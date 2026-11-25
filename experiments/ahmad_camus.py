# ═══════════════════════════════════════════════════════════════
# ahmad_camus.py — Reimplementación de Ahmad et al. (2022) en CAMUS
#
# W. Ahmad, H. Ali, Z. Shah, S. Azmat, "A new generative adversarial
# network for medical images super resolution", Sci. Rep. 12:9533, 2022.
#
# Mismo split por paciente, mismos parches, misma selección de modelo
# y misma evaluación sobre imagen completa que camus_eusr.py, para
# compararse con EUSR-GAN v6 EN LOS MISMOS DATOS. Si encuentra los
# resultados de EUSR-GAN (mismo HR_MODE), hace la prueba de Wilcoxon
# pareada imagen por imagen.
#
# Fiel al artículo:
#   · Parte 1: ramas K3/K5/K7, 2 bloques (conv-BN-ReLU-conv-BN) de 64
#     canales con skip bloque1→bloque2, concatenación → 192 canales
#   · Parte 2: 17 bloques residuales de 192 canales, conv final 3×3
#     sumada a la salida de la Parte 1, sub-pixel ×2
#   · Parte 3: mini-red de 3 bloques residuales de 64 canales, conv
#     192 + BN sumada a la salida de la Parte 2, sub-pixel ×2
#   · Discriminador hasta 2048 canales con rama de 3 capas 128/256/512
#   · Pérdida: L1 + L2 + VGG + adversarial (Ec. 5)
#   · Inicialización normal; BN con gamma ~ N(1, 0.02)
#
# Supuestos (el artículo no los especifica):
#   · Pesos de la Ec. 5: VGG ×0.006 y adversarial ×1e-3, la escala de
#     SRGAN (Ledig et al. 2017), del que parte el artículo. Con pesos 1
#     la pérdida adversarial domina y el entrenamiento no converge.
#   · Optimizador Adam; preentrenamiento con L1+L2 antes del GAN (como
#     SRGAN) y el mismo calendario/selección por SSIM de val que EUSR-GAN.
#   · Desviación de la inicialización 0.02 (la Tabla 1 dice 0.2, que
#     tomamos como errata); PReLU tras cada sub-pixel; reconstrucción
#     conv 3×3 con sigmoide (imágenes en [0, 1]).
#   · Capas del discriminador que bajan de 2048 a 512 con kernel 1×1.
#   · Batch 4 como EUSR-GAN (el artículo usa 2 en CAMUS).
#
# Requisitos:  pip install -c constraints.txt nibabel
# Uso:  %run ahmad_camus.py  (celda)  o  python ahmad_camus.py  (terminal)
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
    Conv2D, BatchNormalization, ReLU, PReLU, LeakyReLU, Add, Concatenate,
    Input, Dense, Flatten, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam

# ── Datos y protocolo (IDÉNTICOS a camus_eusr.py) ─────────────
CAMUS_DIR = 'camus/database_nifti'
SEED      = 42
VIEWS, PHASES = ['4CH'], ['ED']
HR_MODE   = 'ahmad'       # 'ahmad' (1024×512) | 'native'
HR_H, HR_W = 1024, 512
N_TEST, N_VAL = 100, 40
PATCHES_PER_IMG = 10
VAL_PATCHES     = 4
MIN_SECTOR      = 0.5
FREE_FRAC       = 0.25
TILE_STRIDE     = 16

OUT_DIR = 'resultados_ahmad_camus' + ('' if HR_MODE == 'ahmad' else '_native')
EUSR_RESULTS = ('resultados_camus' + ('' if HR_MODE == 'ahmad' else '_native')
                + '/resultados_camus.json')

# ── Entrenamiento ─────────────────────────────────────────────
SCALE, HR_PATCH = 4, 128
LR_PATCH   = HR_PATCH // SCALE
BATCH_SIZE = 4
EPOCHS_PRE, EPOCHS_GAN = 50, 30
RUN_GAN    = True         # False = solo preentrenamiento (L1+L2)
LR_PRE, LR_G, LR_D = 1e-4, 1e-5, 1e-5
PATIENCE, EARLY_STOP = 8, 16
INIT_STD   = 0.02
W_L1, W_L2, W_VGG, W_GEN = 1.0, 1.0, 0.006, 1e-3
INFER      = 'full'       # 'full' (imagen completa) | 'tiles' (como EUSR-GAN)

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
W_PRE = os.path.join(OUT_DIR, 'ahmad_pre_best.weights.h5')
W_GAN = os.path.join(OUT_DIR, 'ahmad_gan_best.weights.h5')
P_HR  = HR_PATCH


def predict_sr(G, lr, batch: int = 16) -> np.ndarray:
    return np.concatenate([G(lr[i:i + batch], training=False).numpy()
                           for i in range(0, len(lr), batch)])


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


def psnr_per_image(y_true, y_pred) -> np.ndarray:
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    return (10.0 * tf.math.log(1.0 / (mse + 1e-10)) /
            tf.math.log(10.0)).numpy()


def ssim_per_image(y_true, y_pred) -> np.ndarray:
    return tf.image.ssim(y_true, y_pred, max_val=1.0).numpy()


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


def load_set(items):
    raw = [load_nifti(it['img']) for it in items]
    hrs = [to_hr(a) for a in raw]
    return {'items': items, 'raw_shapes': [a.shape for a in raw], 'hr': hrs,
            'mask': [sector_mask(h) for h in hrs]}


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

# ═══════════════════════════════════════════════════════════════
# GENERADOR DE AHMAD ET AL.
# ═══════════════════════════════════════════════════════════════
def _init():
    return tf.keras.initializers.RandomNormal(0.0, INIT_STD)


def _bn():
    return BatchNormalization(
        gamma_initializer=tf.keras.initializers.RandomNormal(1.0, 0.02))


def _conv(x, f, k, s=1):
    return Conv2D(f, k, strides=s, padding='same', kernel_initializer=_init())(x)


def _unit(x, f, k):
    """conv-BN-ReLU-conv-BN (bloque del artículo)."""
    y = ReLU()(_bn()(_conv(x, f, k)))
    return _bn()(_conv(y, f, k))


def _branch(x, k):
    b1 = _unit(x, 64, k)
    b2 = _unit(b1, 64, k)
    return Add()([b1, b2])            # skip bloque 1 → bloque 2


def _subpixel(x, f):
    x = _conv(x, f * 4, 3)
    x = Lambda(lambda t: tf.nn.depth_to_space(t, 2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def build_ahmad_generator():
    inp = Input((None, None, 1), name='lr_input')
    # Parte 1 — rasgos superficiales multi-escala → 192 canales
    p1 = Concatenate(name='p1_concat')([_branch(inp, k) for k in (3, 5, 7)])
    # Parte 2 — 17 bloques residuales de 192 canales + ×2
    x = p1
    for _ in range(17):
        x = Add()([x, _unit(x, 192, 3)])
    x = Add()([_bn()(_conv(x, 192, 3)), p1])
    up2 = _subpixel(x, 192)
    # Parte 3 — mini-red de 3 bloques de 64 canales + ×2
    m = _conv(up2, 64, 3)
    for _ in range(3):
        m = Add()([m, _unit(m, 64, 3)])
    x = Add()([up2, _bn()(_conv(m, 192, 3))])
    x = _subpixel(x, 192)
    out = Conv2D(1, 3, padding='same', activation='sigmoid',
                 kernel_initializer=_init(), name='hr_output')(x)
    G = Model(inp, out, name='Ahmad2022_G')
    print(f"Generador Ahmad et al. | parámetros: {G.count_params():,}")
    return G


# ═══════════════════════════════════════════════════════════════
# DISCRIMINADOR DE AHMAD ET AL. (hasta 2048 canales + rama con skip)
# ═══════════════════════════════════════════════════════════════
def _dconv(x, f, s=1, k=3, bn=True):
    x = _conv(x, f, k, s)
    if bn:
        x = _bn()(x)
    return LeakyReLU(0.2)(x)


def build_ahmad_discriminator(size=HR_PATCH):
    inp = Input((size, size, 1))
    x = _dconv(inp, 64, 1, bn=False)
    for f, s in ((64, 2), (128, 1), (128, 2), (256, 1),     # SRGAN: 8 capas
                 (256, 2), (512, 1), (512, 2)):
        x = _dconv(x, f, s)
    x = _dconv(x, 1024, 2)                                 # sube hasta 2048
    x = _dconv(x, 2048, 1)
    x = _dconv(x, 1024, 1, k=1)                            # baja hasta 512
    x0 = _dconv(x, 512, 1, k=1)
    y = _dconv(x0, 128, 1, k=1)                            # rama 128/256/512
    y = _dconv(y, 256, 1)
    y = _dconv(y, 512, 1)
    x = Add()([x0, y])
    x = LeakyReLU(0.2)(Dense(1024)(Flatten()(x)))
    out = Dense(1, activation='sigmoid')(x)
    D = Model(inp, out, name='Ahmad2022_D')
    print(f"Discriminador Ahmad et al. | parámetros: {D.count_params():,}")
    return D


# ═══════════════════════════════════════════════════════════════
# PÉRDIDAS (Ec. 1–5)
# ═══════════════════════════════════════════════════════════════
_vgg = None


def vgg_loss(hr, sr):
    global _vgg
    if _vgg is None:
        base = VGG19(weights='imagenet', include_top=False,
                     input_shape=(HR_PATCH, HR_PATCH, 3))
        _vgg = Model(base.input, base.get_layer('block5_conv4').output)
        _vgg.trainable = False
    return tf.reduce_mean(tf.square(_vgg(tf.image.grayscale_to_rgb(hr)) -
                                    _vgg(tf.image.grayscale_to_rgb(sr))))


def pixel_loss(hr, sr):
    return (W_L1 * tf.reduce_mean(tf.abs(hr - sr)) +
            W_L2 * tf.reduce_mean(tf.square(hr - sr)))


_bce = tf.keras.losses.BinaryCrossentropy()


# ═══════════════════════════════════════════════════════════════
# ENTRENAMIENTO (mismo calendario y selección que EUSR-GAN)
# ═══════════════════════════════════════════════════════════════
def evaluate_patches(G, lr, hr):
    sr = predict_sr(G, lr)
    return (float(np.mean(psnr_per_image(hr, sr))),
            float(np.mean(ssim_per_image(hr, sr))))


def pretrain(G, x_tr, y_tr, x_va, y_va):
    print("═" * 60 + "\nPREENTRENAMIENTO — L1 + L2\n" + "═" * 60)
    lr_now, opt = LR_PRE, Adam(LR_PRE, beta_1=0.9)
    best, no_imp, no_imp_tot = -1.0, 0, 0
    n_b = len(x_tr) // BATCH_SIZE

    @tf.function
    def step(lr_b, hr_b):
        with tf.GradientTape() as t:
            loss = pixel_loss(hr_b, G(lr_b, training=True))
        g, _ = tf.clip_by_global_norm(t.gradient(loss, G.trainable_variables), 1.0)
        opt.apply_gradients(zip(g, G.trainable_variables))
        return loss

    for ep in range(1, EPOCHS_PRE + 1):
        t0, idx, ls = time.time(), np.random.permutation(len(x_tr)), []
        for i in range(n_b):
            b = idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            ls.append(step(x_tr[b], y_tr[b]).numpy())
        p, s = evaluate_patches(G, x_va, y_va)
        mejora = s > best
        if mejora:
            best, no_imp, no_imp_tot = s, 0, 0
            G.save_weights(W_PRE)
        else:
            no_imp += 1
            no_imp_tot += 1
        if no_imp >= PATIENCE and lr_now > 1e-5:
            lr_now, no_imp = max(lr_now * 0.5, 1e-5), 0
            opt.learning_rate.assign(lr_now)
            print(f"  ↓ lr → {lr_now:.2e}")
        print(f"  Época {ep:03d}/{EPOCHS_PRE} | loss: {np.mean(ls):.4f} | PSNR: {p:.2f} dB | "
              f"SSIM: {s:.4f} | {time.time() - t0:.0f}s" + (" ← guardado" if mejora else ""),
              flush=True)
        if no_imp_tot >= EARLY_STOP:
            print(f"\n  Sin mejora en {EARLY_STOP} épocas — fin del preentrenamiento.")
            break
    G.load_weights(W_PRE)
    return best


def train_gan(G, D, x_tr, y_tr, x_va, y_va, best_pre):
    print("═" * 60 + "\nGAN — L1 + L2 + VGG + adversarial (Ec. 5)\n" + "═" * 60)
    opt_g, opt_d = Adam(LR_G, beta_1=0.9), Adam(LR_D, beta_1=0.9)
    best, no_imp, saved = best_pre, 0, False
    n_b = len(x_tr) // BATCH_SIZE

    @tf.function
    def step(lr_b, hr_b):
        with tf.GradientTape() as tg, tf.GradientTape() as td:
            sr = G(lr_b, training=True)
            d_real, d_fake = D(hr_b, training=True), D(sr, training=True)
            l_gen = tf.reduce_mean(-tf.math.log(d_fake + 1e-8))
            lg = pixel_loss(hr_b, sr) + W_VGG * vgg_loss(hr_b, sr) + W_GEN * l_gen
            ld = _bce(tf.ones_like(d_real), d_real) + _bce(tf.zeros_like(d_fake), d_fake)
        gg, _ = tf.clip_by_global_norm(tg.gradient(lg, G.trainable_variables), 1.0)
        gd, _ = tf.clip_by_global_norm(td.gradient(ld, D.trainable_variables), 1.0)
        opt_g.apply_gradients(zip(gg, G.trainable_variables))
        opt_d.apply_gradients(zip(gd, D.trainable_variables))
        return lg, ld

    for ep in range(1, EPOCHS_GAN + 1):
        t0, idx, gl, dl = time.time(), np.random.permutation(len(x_tr)), [], []
        for i in range(n_b):
            b = idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            lg, ld = step(x_tr[b], y_tr[b])
            gl.append(lg.numpy()); dl.append(ld.numpy())
        p, s = evaluate_patches(G, x_va, y_va)
        mejora = s > best
        if mejora:
            best, no_imp, saved = s, 0, True
            G.save_weights(W_GAN)
        else:
            no_imp += 1
        print(f"  [GAN] Época {ep:03d}/{EPOCHS_GAN} | G: {np.mean(gl):.4f} | D: {np.mean(dl):.4f} | "
              f"PSNR: {p:.2f} dB | SSIM: {s:.4f} | {time.time() - t0:.0f}s"
              + (" ← guardado" if mejora else ""), flush=True)
        if no_imp >= PATIENCE:
            print(f"\n  Sin mejora en {PATIENCE} épocas — fin del GAN.")
            break
    G.load_weights(W_GAN if saved else W_PRE)
    print("GAN mejoró el preentrenamiento" if saved else
          "GAN no superó el preentrenamiento → se conservan esos pesos")
    return saved


# ═══════════════════════════════════════════════════════════════
# EVALUACIÓN EN TEST (imagen completa) Y COMPARACIÓN PAREADA
# ═══════════════════════════════════════════════════════════════
def sr_full(G, lr):
    return np.clip(G(lr[None, ..., None], training=False).numpy()[0, ..., 0], 0, 1)


def evaluate_test(G, test):
    rows = []
    for it, hr in zip(test['items'], test['hr']):
        lr  = degrade(hr)
        bic = bicubic_full(lr, hr.shape)
        sr  = sr_full(G, lr) if INFER == 'full' else sr_tiles(G, lr)
        m   = sector_mask(hr)
        r = {'patient': it['patient'], 'quality': it['quality']}
        for name, img in (('bic', bic), ('sr', sr)):
            r[f'{name}_psnr']        = psnr(hr, img)
            r[f'{name}_ssim']        = ssim_full(hr, img)
            r[f'{name}_psnr_sector'] = psnr(hr, img, m)
            r[f'{name}_ssim_sector'] = float(ssim_map(hr, img)[m].mean())
        rows.append(r)
    return rows


def _ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.2f} ± {x.std():.2f}" if x.mean() > 1 else f"{x.mean():.3f} ± {x.std():.3f}"


def report(rows, gan_used):
    print("\n" + "═" * 72)
    print(f"CAMUS — TEST ({len(rows)} imágenes, ×{SCALE}, HR = {HR_MODE}, inferencia = {INFER})")
    print("═" * 72)
    eusr = None
    if os.path.exists(EUSR_RESULTS):
        e = json.load(open(EUSR_RESULTS))
        if e.get('config', {}).get('hr_mode') == HR_MODE:
            eusr = {r['patient']: r for r in e['por_imagen']}
    for zona, suf in (('Imagen completa', ''), ('Solo sector', '_sector')):
        print(f"\n{zona}:")
        print(f"  {'Bicúbica':26s} PSNR {_ms([r['bic_psnr' + suf] for r in rows])}   "
              f"SSIM {_ms([r['bic_ssim' + suf] for r in rows])}")
        print(f"  {'Ahmad et al. (reimpl.)':26s} PSNR {_ms([r['sr_psnr' + suf] for r in rows])}   "
              f"SSIM {_ms([r['sr_ssim' + suf] for r in rows])}")
        if eusr:
            com = [r for r in rows if r['patient'] in eusr]
            ep  = [eusr[r['patient']]['sr_psnr' + suf] for r in com]
            es  = [eusr[r['patient']]['sr_ssim' + suf] for r in com]
            ap  = [r['sr_psnr' + suf] for r in com]
            as_ = [r['sr_ssim' + suf] for r in com]
            print(f"  {'EUSR-GAN v6':26s} PSNR {_ms(ep)}   SSIM {_ms(es)}")
            for nom, a, b in (('PSNR', ep, ap), ('SSIM', es, as_)):
                d = np.array(a) - np.array(b)
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = float('nan')
                print(f"    EUSR − Ahmad {nom}: {d.mean():+.4f}  "
                      f"(mejor en {np.mean(d > 0):.0%} de {len(d)} imágenes, p Wilcoxon = {p:.1e})")
    if not eusr:
        print(f"\n(No se encontró {EUSR_RESULTS} con HR_MODE='{HR_MODE}'; "
              "sin comparación pareada con EUSR-GAN.)")
    print(f"\nReportado por Ahmad et al. (CAMUS): 37.11 ± 1.7 dB / 0.95 · GAN usado: {gan_used}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    items = list_camus()
    tr_i, va_i, te_i = split_patients(items)
    print(f"CAMUS: {len(items)} imágenes | train {len(tr_i)}  val {len(va_i)}  test {len(te_i)}")
    sets = {k: load_set(v) for k, v in (('train', tr_i), ('val', va_i), ('test', te_i))}
    x_tr, y_tr = build_patch_set(sets['train']['hr'], sets['train']['mask'],
                                 PATCHES_PER_IMG, SEED, do_aug=True)
    x_va, y_va = build_patch_set(sets['val']['hr'], sets['val']['mask'],
                                 VAL_PATCHES, SEED + 1, do_aug=False)
    print(f"Parches → train {len(x_tr)}  val {len(x_va)}")

    G = build_ahmad_generator()
    best = pretrain(G, x_tr, y_tr, x_va, y_va)
    gan_used = False
    if RUN_GAN:
        D = build_ahmad_discriminator()
        gan_used = train_gan(G, D, x_tr, y_tr, x_va, y_va, best)

    print("\nEvaluando test sobre imágenes completas…")
    rows = evaluate_test(G, sets['test'])
    report(rows, gan_used)
    with open(os.path.join(OUT_DIR, 'resultados_ahmad.json'), 'w') as fh:
        json.dump({'config': {'hr_mode': HR_MODE, 'infer': INFER, 'gan_used': gan_used,
                              'weights': [W_L1, W_L2, W_VGG, W_GEN], 'batch': BATCH_SIZE},
                   'por_imagen': rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min → {OUT_DIR}/")


if __name__ == '__main__':
    main()
