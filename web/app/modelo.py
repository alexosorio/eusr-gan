# ═══════════════════════════════════════════════════════════════
# modelo.py — EUSR-GAN para el prototipo web
#
# La arquitectura es una copia LITERAL de build_generator de
# busi_native.py / camus_eusr.py (v6, 4 RRDB, salida sigmoide), de modo
# que los pesos .weights.h5 entrenados se cargan sin conversión.
# No importa busi_native.py para no arrastrar VGG19 ni el entrenamiento.
#
# La inferencia es la misma del artículo: mosaicos LR de 32×32 con paso
# 16, unidos con ventana de Hann; opcionalmente, autoensamble ×8.
# ═══════════════════════════════════════════════════════════════
import os
import threading

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, GroupNormalization, ReLU,
    Add, Multiply, Concatenate, Input, Dense,
    GlobalAveragePooling2D, Lambda, Reshape
)
from tensorflow.keras.models import Model

SCALE = 4
LR_PATCH = 32
HR_PATCH = LR_PATCH * SCALE
TILE_STRIDE = 16
GN_GROUPS = 8
GROWTH_CH = 32
CH = 64
NUM_RRDB = 4
LOTE = int(os.environ.get('LOTE', '16'))      # mosaicos por lote: menos lote = menos memoria


# ── Bloques (idénticos a busi_native.py) ──────────────────────
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
    ch = x.shape[-1]
    mid = max(ch // reduction, 8)
    s = GlobalAveragePooling2D()(x)
    s = Dense(mid, activation='relu')(s)
    s = Dense(ch, activation='sigmoid')(s)
    s = Reshape((1, 1, ch))(s)
    return Multiply()([x, s])


def pixel_shuffle(x, scale: int = 2):
    ch = x.shape[-1]
    x = Conv2D(ch * scale * scale, 3, padding='same', use_bias=False)(x)
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
    x = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    x = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    x = Add()([x, Lambda(lambda t: t * beta)(dense_block(x))])
    return Add()([res, Lambda(lambda t: t * beta)(x)])


def build_generator(lr_size: int = LR_PATCH, n_rrdb: int = NUM_RRDB, ch: int = CH):
    inp = Input(shape=(lr_size, lr_size, 1), name='lr_input')

    # Parte 1 — extracción multiescala (K3 / K5 separable / K7 dilatada)
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

    # Parte 2 — tronco RRDB + ×2
    p2 = gn_relu(Conv2D(ch, 1, use_bias=False)(p1))
    p2_in = p2
    for _ in range(n_rrdb):
        p2 = rrdb_block(p2)
    p2 = Add()([Conv2D(ch, 3, padding='same', use_bias=False)(p2), p2_in])
    p2 = gn_relu(pixel_shuffle(p2, scale=2))

    # Parte 3 — mini-red + salto denso desde la Parte 1 + ×2
    p3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p2))
    p3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p3))
    skip_p1 = Lambda(
        lambda t: tf.image.resize(t, (lr_size * 2, lr_size * 2), method='bilinear'),
        name='skip_p1_upsample')(Conv2D(ch, 1, use_bias=False)(p1))
    p3 = Add(name='skip_dense_add')([p3, skip_p1])
    p3 = gn_relu(pixel_shuffle(p3, scale=2))
    p3 = gn_relu(Conv2D(ch, 3, padding='same', use_bias=False)(p3))
    out = Conv2D(1, 3, padding='same', activation='sigmoid', name='hr_output')(p3)
    return Model(inputs=inp, outputs=out, name='EUSR_GAN_v6')


# ── Inferencia ────────────────────────────────────────────────
class Modelo:
    """Generador cargado una sola vez; la inferencia es segura entre hilos."""

    def __init__(self, ruta_pesos: str):
        self.G = build_generator()
        self.G.load_weights(ruta_pesos)
        self.ruta = ruta_pesos
        self._lock = threading.Lock()          # una inferencia a la vez por modelo
        # grafo compilado: usa mucha menos memoria y es más rápido que el modo eager
        self._f = tf.function(lambda x: self.G(x, training=False),
                              input_signature=[tf.TensorSpec([None, LR_PATCH, LR_PATCH, 1], tf.float32)])
        self._f(tf.zeros((1, LR_PATCH, LR_PATCH, 1)))   # compila y calienta

    def _predict(self, tiles: np.ndarray, batch: int = LOTE) -> np.ndarray:
        return np.concatenate([self._f(tf.constant(tiles[i:i + batch])).numpy()
                               for i in range(0, len(tiles), batch)])

    def sr_tiles(self, lr: np.ndarray, stride: int = TILE_STRIDE) -> np.ndarray:
        """Igual que sr_tiles de busi_native.py: mosaicos 32×32 y ventana de Hann."""
        t, T = LR_PATCH, HR_PATCH
        h, w = lr.shape
        w1 = np.hanning(T + 2)[1:-1]
        win = np.outer(w1, w1).astype(np.float32)
        coords = [(y, x) for y in posiciones(h, t, stride) for x in posiciones(w, t, stride)]
        tiles = np.stack([lr[y:y + t, x:x + t] for y, x in coords])[..., None]
        pred = self._predict(tiles)[..., 0]
        out = np.zeros((h * SCALE, w * SCALE), np.float32)
        acc = np.zeros_like(out)
        for (y, x), p in zip(coords, pred):
            Y, X = y * SCALE, x * SCALE
            out[Y:Y + T, X:X + T] += p * win
            acc[Y:Y + T, X:X + T] += win
        return np.clip(out / acc, 0.0, 1.0)

    def super_resolver(self, lr: np.ndarray, autoensamble: bool = False) -> np.ndarray:
        """LR (H, W) en [0, 1], H y W ≥ 32 → SR (4H, 4W)."""
        lr = np.ascontiguousarray(lr, dtype=np.float32)
        with self._lock:
            if not autoensamble:
                return self.sr_tiles(lr)
            acc = None
            for k in range(4):
                for flip in (False, True):
                    x = np.rot90(lr, k)
                    x = x[:, ::-1] if flip else x
                    y = self.sr_tiles(np.ascontiguousarray(x))
                    y = y[:, ::-1] if flip else y
                    y = np.rot90(y, -k)
                    acc = y if acc is None else acc + y
            return np.clip(acc / 8.0, 0.0, 1.0)


def posiciones(n: int, t: int, stride: int) -> list:
    p = list(range(0, n - t + 1, stride))
    if p[-1] != n - t:
        p.append(n - t)
    return p


def n_mosaicos(h: int, w: int) -> int:
    return len(posiciones(h, LR_PATCH, TILE_STRIDE)) * len(posiciones(w, LR_PATCH, TILE_STRIDE))


# ── Degradación, interpolación y métricas (como en el artículo) ─
def degradar(hr: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """LR por reducción bicúbica ×4 con PIL, igual que en el entrenamiento."""
    im = Image.fromarray(np.clip(hr * 255, 0, 255).astype(np.uint8))
    lr = im.resize((hr.shape[1] // scale, hr.shape[0] // scale), Image.BICUBIC)
    return np.asarray(lr, dtype=np.float32) / 255.0


def bicubica(lr: np.ndarray, shape) -> np.ndarray:
    return np.clip(tf.image.resize(lr[None, ..., None], shape,
                                   method='bicubic').numpy()[0, ..., 0], 0, 1)


def vecino(lr: np.ndarray, scale: int = SCALE) -> np.ndarray:
    return np.repeat(np.repeat(lr, scale, axis=0), scale, axis=1)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(tf.image.ssim(a[None, ..., None], b[None, ..., None], max_val=1.0)[0])
