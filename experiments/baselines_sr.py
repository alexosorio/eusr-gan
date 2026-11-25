# ═══════════════════════════════════════════════════════════════
# baselines_sr.py — Métodos de comparación para EUSR-GAN (Keras, TF 2.12)
#
#   EDSR-baseline  (Lim et al., CVPRW 2017): 16 bloques residuales,
#                  64 canales, sin normalización, ×4 con dos sub-pixel ×2.
#   SwinIR-light   (Liang et al., ICCVW 2021, versión "lightweight"):
#                  4 RSTB × 6 bloques Swin, dim 60, 6 cabezas, ventana 8,
#                  mlp_ratio 2, upsampler 'pixelshuffledirect'.
#
# Ambos reciben LR 32×32×1 en [0, 1] y devuelven HR 128×128×1, igual que
# EUSR-GAN, así que se entrenan y evalúan con el MISMO pipeline (parches,
# pérdidas, optimizador, parada temprana e inferencia por mosaicos).
# Configuraciones según los artículos originales; solo cambia la entrada
# a 1 canal (escala de grises) y no hay resta de la media RGB.
# ═══════════════════════════════════════════════════════════════
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# ───────────────────────────── EDSR ─────────────────────────────
def build_edsr(lr_size=32, n_resblocks=16, n_feats=64, res_scale=1.0, scale=4):
    inp = layers.Input(shape=(lr_size, lr_size, 1), name='lr_input')
    x = layers.Conv2D(n_feats, 3, padding='same', name='head')(inp)
    h = x
    for i in range(n_resblocks):
        r = layers.Conv2D(n_feats, 3, padding='same', activation='relu', name=f'rb{i}_c1')(h)
        r = layers.Conv2D(n_feats, 3, padding='same', name=f'rb{i}_c2')(r)
        if res_scale != 1.0:
            r = layers.Lambda(lambda t: t * res_scale)(r)
        h = layers.Add(name=f'rb{i}_add')([h, r])
    h = layers.Conv2D(n_feats, 3, padding='same', name='body_tail')(h)
    h = layers.Add(name='global_skip')([h, x])
    for k in range(int(np.log2(scale))):
        h = layers.Conv2D(4 * n_feats, 3, padding='same', name=f'up{k}_conv')(h)
        h = layers.Lambda(lambda t: tf.nn.depth_to_space(t, 2), name=f'up{k}_ps')(h)
    out = layers.Conv2D(1, 3, padding='same', name='hr_output')(h)
    model = Model(inp, out, name='EDSR_baseline')
    print(f"EDSR-baseline | parámetros: {model.count_params():,}")
    return model


# ──────────────────────────── SwinIR ────────────────────────────
def _window_partition(x, ws):
    # x: (B, H, W, C) → (B·nW, ws·ws, C)
    B = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, (B, H // ws, ws, W // ws, ws, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, ws * ws, C))


def _window_reverse(w, ws, H, W):
    C = w.shape[-1]
    x = tf.reshape(w, (-1, H // ws, W // ws, ws, ws, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, H, W, C))


class SwinBlock(layers.Layer):
    """Bloque Swin con atención por ventanas (W-MSA / SW-MSA) y sesgo de
    posición relativa, como en SwinIR. Resolución fija (H, W)."""

    def __init__(self, dim, heads, H, W, window=8, shift=0, mlp_ratio=2.0, **kw):
        super().__init__(**kw)
        self.dim, self.heads, self.H, self.W = dim, heads, H, W
        self.ws = min(window, H, W)
        self.shift = 0 if min(H, W) <= window else shift
        self.scale = (dim // heads) ** -0.5
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.qkv = layers.Dense(3 * dim)
        self.proj = layers.Dense(dim)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = layers.Dense(int(dim * mlp_ratio))
        self.fc2 = layers.Dense(dim)

        ws = self.ws
        coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing='ij')).reshape(2, -1)
        rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (ws - 1)
        self.rel_index = tf.constant(rel[..., 0] * (2 * ws - 1) + rel[..., 1], tf.int32)

        if self.shift > 0:
            img = np.zeros((1, H, W, 1), np.float32)
            cnt = 0
            for hs in (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None)):
                for wsl in (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None)):
                    img[:, hs, wsl, :] = cnt
                    cnt += 1
            mw = _window_partition(tf.constant(img), ws)[..., 0].numpy()   # (nW, N)
            m = mw[:, None, :] - mw[:, :, None]
            self.attn_mask = tf.constant(np.where(m != 0, -100.0, 0.0), tf.float32)
        else:
            self.attn_mask = None

    def build(self, input_shape):
        ws = self.ws
        self.rel_table = self.add_weight(
            name='relative_position_bias_table', shape=((2 * ws - 1) ** 2, self.heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (B, H·W, C)
        H, W, C, ws, nh = self.H, self.W, self.dim, self.ws, self.heads
        B = tf.shape(x)[0]
        shortcut = x
        h = tf.reshape(self.norm1(x), (B, H, W, C))
        if self.shift > 0:
            h = tf.roll(h, shift=(-self.shift, -self.shift), axis=(1, 2))
        win = _window_partition(h, ws)                                # (B·nW, N, C)
        N = ws * ws
        qkv = tf.reshape(self.qkv(win), (-1, N, 3, nh, C // nh))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))                      # (3, B·nW, nh, N, d)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True)                      # (B·nW, nh, N, N)
        bias = tf.gather(self.rel_table, tf.reshape(self.rel_index, (-1,)))
        bias = tf.transpose(tf.reshape(bias, (N, N, nh)), (2, 0, 1))
        attn = attn + bias[None]
        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            attn = tf.reshape(attn, (-1, nW, nh, N, N)) + self.attn_mask[None, :, None]
            attn = tf.reshape(attn, (-1, nh, N, N))
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3))
        out = self.proj(tf.reshape(out, (-1, N, C)))
        h = _window_reverse(out, ws, H, W)
        if self.shift > 0:
            h = tf.roll(h, shift=(self.shift, self.shift), axis=(1, 2))
        x = shortcut + tf.reshape(h, (B, H * W, C))
        y = self.fc2(tf.nn.gelu(self.fc1(self.norm2(x)), approximate=False))
        return x + y


def build_swinir_light(lr_size=32, embed_dim=60, depths=(6, 6, 6, 6),
                       heads=(6, 6, 6, 6), window=8, mlp_ratio=2.0, scale=4):
    H = W = lr_size
    C = embed_dim
    inp = layers.Input(shape=(H, W, 1), name='lr_input')
    x0 = layers.Conv2D(C, 3, padding='same', name='conv_first')(inp)
    to_seq = layers.Reshape((H * W, C))
    to_img = layers.Reshape((H, W, C))
    t = layers.LayerNormalization(epsilon=1e-5, name='patch_embed_norm')(to_seq(x0))
    for i, (d, nh) in enumerate(zip(depths, heads)):
        res_in = t
        for j in range(d):
            t = SwinBlock(C, nh, H, W, window, shift=0 if j % 2 == 0 else window // 2,
                          mlp_ratio=mlp_ratio, name=f'rstb{i}_blk{j}')(t)
        # RSTB: conv 3×3 sobre la imagen y conexión residual del grupo
        c = layers.Conv2D(C, 3, padding='same', name=f'rstb{i}_conv')(to_img(t))
        t = layers.Add(name=f'rstb{i}_add')([layers.Reshape((H * W, C))(c), res_in])
    t = layers.LayerNormalization(epsilon=1e-5, name='norm')(t)
    x = layers.Conv2D(C, 3, padding='same', name='conv_after_body')(layers.Reshape((H, W, C))(t))
    x = layers.Add(name='body_skip')([x, x0])
    # pixelshuffledirect (versión ligera de SwinIR)
    x = layers.Conv2D(scale * scale, 3, padding='same', name='upsample_conv')(x)
    out = layers.Lambda(lambda z: tf.nn.depth_to_space(z, scale), name='hr_output')(x)
    model = Model(inp, out, name='SwinIR_light')
    print(f"SwinIR-light | parámetros: {model.count_params():,}")
    return model


BUILDERS = {'edsr': build_edsr, 'swinir': build_swinir_light}
NAMES = {'edsr': 'EDSR-baseline', 'swinir': 'SwinIR-light'}
