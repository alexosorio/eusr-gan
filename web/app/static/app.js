// ═══════════════════════════════════════════════════════════════
// app.js — interfaz del prototipo EUSR-GAN (sin dependencias)
// ═══════════════════════════════════════════════════════════════
'use strict';

const $ = (id) => document.getElementById(id);
const S = {
  servidor: null,      // respuesta de /api/estado
  archivo: null,       // File elegido
  img: null,           // vista previa (HTMLImageElement)
  ancho: 0, alto: 0,   // dimensiones de la imagen en píxeles
  zona: null,          // {x, y, w, h} en píxeles de la imagen, o null = completa
  arrastre: null,      // estado del puntero sobre el lienzo
  resultado: null,
  pos: 50,             // posición del deslizador (%)
};

const DESC_MODO = {
  ampliar: 'Amplía ×4 la zona tal como fue adquirida. Uso exploratorio: el modelo se entrenó para ' +
           'reconstruir imágenes reducidas ×4, así que aquí no hay referencia para medir la mejora.',
  validar: 'Reduce la zona ×4 y la reconstruye; la zona original sirve de referencia para medir PSNR ' +
           'y SSIM. Es el escenario evaluado en el artículo.',
};
const NOMBRES = {
  entrada: 'Entrada (píxeles)', bicubica: 'Interpolación bicúbica', eusr: 'EUSR-GAN',
  referencia: 'Referencia original', error_bicubica: 'Error · bicúbica', error_eusr: 'Error · EUSR-GAN',
};

const num = (v, d = 2) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
const modo = () => document.querySelector('input[name="modo"]:checked').value;
const vista = () => document.querySelector('input[name="vista"]:checked').value;

function mostrarError(msg) { $('error').textContent = msg || ''; }

// ── Estado del servidor ───────────────────────────────────────
async function iniciar() {
  try {
    const r = await fetch('/api/estado');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    S.servidor = await r.json();
  } catch (e) {
    $('estado-servidor').textContent = 'Servidor no disponible';
    $('estado-servidor').className = 'pastilla mal';
    return;
  }
  const sel = $('modelo');
  sel.innerHTML = '';
  for (const m of S.servidor.modelos) {
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.nombre; o.dataset.desc = m.descripcion;
    sel.appendChild(o);
  }
  const n = S.servidor.modelos.length;
  if (n) {
    $('estado-servidor').textContent = n === 1 ? `${S.servidor.modelos[0].nombre} · listo` : `${n} modelos listos`;
    $('estado-servidor').className = 'pastilla ok';
  } else {
    $('estado-servidor').textContent = 'Sin modelos cargados';
    $('estado-servidor').className = 'pastilla mal';
    sel.innerHTML = '<option>No hay pesos en el servidor</option>';
    sel.disabled = true;
  }
  actualizarTextos();
}

function actualizarTextos() {
  const o = $('modelo').selectedOptions[0];
  $('desc-modelo').textContent = o && o.dataset.desc ? `Entrenado con: ${o.dataset.desc}.` : '';
  $('desc-modo').textContent = DESC_MODO[modo()];
  actualizarZona();
}

// ── Carga de la imagen ────────────────────────────────────────
async function cargarArchivo(f) {
  if (!f) return;
  mostrarError('');
  const fd = new FormData();
  fd.append('archivo', f);
  ocupado(true, 'Leyendo la imagen…');
  try {
    const r = await fetch('/api/vista', { method: 'POST', body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `Error ${r.status}`);
    const img = new Image();
    await new Promise((ok, mal) => { img.onload = ok; img.onerror = mal; img.src = d.png; });
    Object.assign(S, { archivo: f, img, ancho: d.ancho, alto: d.alto, zona: null, resultado: null });
    $('info-archivo').textContent = `${f.name} · ${d.ancho}×${d.alto} px`;
    $('dim-imagen').textContent = `${d.ancho}×${d.alto} px`;
    $('vacio').hidden = true;
    $('resultado').hidden = true;
    const c = $('lienzo');
    c.width = d.ancho; c.height = d.alto;
    dibujar();
    actualizarZona();
  } catch (e) {
    mostrarError(`No se pudo cargar la imagen: ${e.message}`);
  } finally {
    ocupado(false);
  }
}

$('archivo').addEventListener('change', (e) => cargarArchivo(e.target.files[0]));
const caja = $('zona-carga');
['dragenter', 'dragover'].forEach((t) => caja.addEventListener(t, (e) => { e.preventDefault(); caja.classList.add('encima'); }));
['dragleave', 'drop'].forEach((t) => caja.addEventListener(t, (e) => { e.preventDefault(); caja.classList.remove('encima'); }));
caja.addEventListener('drop', (e) => cargarArchivo(e.dataTransfer.files[0]));
caja.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('archivo').click(); } });

// ── Lienzo y selección de la zona ─────────────────────────────
function dibujar() {
  const c = $('lienzo'), ctx = c.getContext('2d');
  if (!S.img) return;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(S.img, 0, 0);
  const z = S.zona;
  if (!z) return;
  const k = c.width / Math.max(1, c.getBoundingClientRect().width);   // píxeles de imagen por píxel de pantalla
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.rect(0, 0, c.width, c.height);
  ctx.rect(z.x, z.y, z.w, z.h);
  ctx.fill('evenodd');
  const excede = modo() === 'ampliar' && Math.max(z.w, z.h) > S.servidor.limites.max_lado_ampliar;
  ctx.strokeStyle = excede ? '#ff6b5e' : '#3fd2c2';
  ctx.lineWidth = 2 * k;
  ctx.strokeRect(z.x, z.y, z.w, z.h);
  const txt = `${z.w}×${z.h}`;
  ctx.font = `${Math.round(13 * k)}px system-ui, sans-serif`;
  const tw = ctx.measureText(txt).width + 10 * k, th = 20 * k;
  const ty = z.y > th + 4 * k ? z.y - th - 3 * k : z.y + 3 * k;
  ctx.fillStyle = excede ? '#ff6b5e' : '#3fd2c2';
  ctx.fillRect(z.x, ty, tw, th);
  ctx.fillStyle = '#0c0e11';
  ctx.fillText(txt, z.x + 5 * k, ty + 14.5 * k);
  ctx.restore();
}

function puntoImagen(e) {
  const c = $('lienzo'), r = c.getBoundingClientRect();
  return {
    x: Math.min(S.ancho, Math.max(0, Math.round((e.clientX - r.left) * c.width / r.width))),
    y: Math.min(S.alto, Math.max(0, Math.round((e.clientY - r.top) * c.height / r.height))),
  };
}

const lienzo = $('lienzo');
lienzo.addEventListener('pointerdown', (e) => {
  if (!S.img) return;
  lienzo.setPointerCapture(e.pointerId);
  const p = puntoImagen(e), z = S.zona;
  if (z && p.x > z.x && p.x < z.x + z.w && p.y > z.y && p.y < z.y + z.h) {
    S.arrastre = { tipo: 'mover', dx: p.x - z.x, dy: p.y - z.y };
  } else {
    S.arrastre = { tipo: 'dibujar', x0: p.x, y0: p.y, previa: z };
    S.zona = { x: p.x, y: p.y, w: 0, h: 0 };
  }
});
lienzo.addEventListener('pointermove', (e) => {
  if (!S.img) return;
  const p = puntoImagen(e), a = S.arrastre, z = S.zona;
  if (!a) {
    lienzo.style.cursor = z && p.x > z.x && p.x < z.x + z.w && p.y > z.y && p.y < z.y + z.h ? 'move' : 'crosshair';
    return;
  }
  if (a.tipo === 'mover') {
    z.x = Math.min(S.ancho - z.w, Math.max(0, p.x - a.dx));
    z.y = Math.min(S.alto - z.h, Math.max(0, p.y - a.dy));
  } else {
    S.zona = { x: Math.min(a.x0, p.x), y: Math.min(a.y0, p.y), w: Math.abs(p.x - a.x0), h: Math.abs(p.y - a.y0) };
  }
  dibujar();
  actualizarZona();
});
const soltar = () => {
  const a = S.arrastre;
  S.arrastre = null;
  if (a && a.tipo === 'dibujar' && S.zona && (S.zona.w < 8 || S.zona.h < 8)) S.zona = a.previa;   // un clic no borra la zona
  dibujar();
  actualizarZona();
};
lienzo.addEventListener('pointerup', soltar);
lienzo.addEventListener('pointercancel', soltar);
window.addEventListener('resize', dibujar);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.zona && !S.arrastre) borrarZona(); });

function borrarZona() { S.zona = null; dibujar(); actualizarZona(); }
$('btn-borrar').addEventListener('click', borrarZona);
$('btn-completa').addEventListener('click', () => { S.zona = { x: 0, y: 0, w: S.ancho, h: S.alto }; dibujar(); actualizarZona(); });

function posiciones(n) {
  const t = 32, s = 16, p = [];
  for (let i = 0; i <= n - t; i += s) p.push(i);
  if (p[p.length - 1] !== n - t) p.push(n - t);
  return p.length;
}

function actualizarZona() {
  const hay = !!S.img;
  $('btn-completa').disabled = !hay;
  $('btn-borrar').disabled = !S.zona;
  const listo = hay && S.servidor && S.servidor.modelos.length;
  if (!hay) { $('btn-mejorar').disabled = true; $('estimacion').textContent = ''; return; }
  const z = S.zona || { x: 0, y: 0, w: S.ancho, h: S.alto };
  $('info-zona').textContent = S.zona
    ? `x ${z.x}, y ${z.y} · ${z.w}×${z.h} px`
    : `Sin zona: se usará la imagen completa (${S.ancho}×${S.alto} px).`;
  const lim = S.servidor ? S.servidor.limites : { max_lado_ampliar: 512, min_lado_validar: 128 };
  let mos, problema = '';
  if (modo() === 'ampliar') {
    mos = posiciones(Math.max(z.h, 32)) * posiciones(Math.max(z.w, 32));
    if (Math.max(z.w, z.h) > lim.max_lado_ampliar) {
      problema = `En modo ampliar la zona puede medir hasta ${lim.max_lado_ampliar} px por lado. Marque una zona más pequeña.`;
    }
    $('btn-mejorar').textContent = 'Ampliar zona ×4';
  } else {
    const m = lim.min_lado_validar;
    if (S.ancho < m || S.alto < m) problema = `Para validar, la imagen debe medir al menos ${m}×${m} px.`;
    const w = Math.max(z.w, m), h = Math.max(z.h, m);
    mos = posiciones(Math.floor(h / 4)) * posiciones(Math.floor(w / 4));
    $('btn-mejorar').textContent = 'Reconstruir y medir';
  }
  if ($('autoensamble').checked) mos *= 8;
  const v = S.servidor && S.servidor.mosaicos_por_segundo;
  $('estimacion').textContent = problema ||
    `≈ ${mos.toLocaleString('es-CO')} mosaicos` + (v ? ` · unos ${Math.max(1, Math.round(mos / v))} s` : '');
  $('estimacion').style.color = problema ? 'var(--error)' : '';
  $('btn-mejorar').disabled = !listo || !!problema;
  S.estimado = v ? mos / v : null;
}

document.querySelectorAll('input[name="modo"]').forEach((r) => r.addEventListener('change', () => { actualizarTextos(); dibujar(); }));
$('modelo').addEventListener('change', actualizarTextos);
$('autoensamble').addEventListener('change', actualizarZona);

// ── Procesamiento ─────────────────────────────────────────────
let reloj = null;
function ocupado(si, texto) {
  $('cargando').hidden = !si;
  clearInterval(reloj);
  if (!si) return;
  $('cargando-texto').textContent = texto || 'Procesando…';
  const t0 = Date.now();
  const tick = () => {
    const s = (Date.now() - t0) / 1000;
    $('cargando-tiempo').textContent = S.estimadoActivo
      ? `${Math.round(s)} s de unos ${Math.max(1, Math.round(S.estimadoActivo))} s estimados`
      : `${Math.round(s)} s`;
  };
  tick();
  reloj = setInterval(tick, 500);
}

$('btn-mejorar').addEventListener('click', async () => {
  mostrarError('');
  const fd = new FormData();
  fd.append('archivo', S.archivo);
  fd.append('modelo', $('modelo').value);
  fd.append('modo', modo());
  fd.append('autoensamble', $('autoensamble').checked ? 'true' : 'false');
  if (S.zona) for (const k of ['x', 'y', 'w', 'h']) fd.append(k, String(S.zona[k]));
  S.estimadoActivo = S.estimado;
  ocupado(true, modo() === 'ampliar' ? 'Ampliando la zona con EUSR-GAN…' : 'Reconstruyendo y midiendo…');
  try {
    const r = await fetch('/api/mejorar', { method: 'POST', body: fd });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `Error ${r.status}`);
    S.resultado = d;
    if (S.zona) { [S.zona.x, S.zona.y, S.zona.w, S.zona.h] = d.zona; dibujar(); actualizarZona(); }
    mostrarResultado();
  } catch (e) {
    mostrarError(e.message);
  } finally {
    S.estimadoActivo = null;
    ocupado(false);
  }
});

// ── Resultados ────────────────────────────────────────────────
function mostrarResultado() {
  const d = S.resultado, I = d.imagenes, validar = d.modo === 'validar';
  $('resultado').hidden = false;

  // selector de la imagen izquierda
  const sel = $('izquierda');
  [...sel.options].forEach((o) => { o.hidden = !(o.value in I); });
  if (!(sel.value in I)) sel.value = 'bicubica';
  $('tab-error').hidden = !validar;
  if (!validar && vista() === 'error') document.querySelector('input[name="vista"][value="deslizador"]').checked = true;

  // avisos y métricas
  $('avisos').innerHTML = '';
  for (const a of d.avisos) { const p = document.createElement('p'); p.textContent = a; $('avisos').appendChild(p); }
  const m = d.metricas;
  $('metricas').hidden = !m;
  if (m) {
    $('metricas').innerHTML = `
      <div class="metrica"><div class="rot">PSNR · bicúbica</div><div class="val">${num(m.psnr_bicubica)} dB</div>
        <div class="sub">SSIM ${num(m.ssim_bicubica, 4)}</div></div>
      <div class="metrica"><div class="rot">PSNR · EUSR-GAN</div><div class="val">${num(m.psnr_eusr)} dB</div>
        <div class="sub">SSIM ${num(m.ssim_eusr, 4)}</div></div>
      <div class="metrica destacada"><div class="rot">Ganancia de EUSR-GAN</div>
        <div class="val">${m.ganancia_psnr >= 0 ? '+' : ''}${num(m.ganancia_psnr)} dB</div>
        <div class="sub">SSIM ${m.ganancia_ssim >= 0 ? '+' : ''}${num(m.ganancia_ssim, 4)}</div></div>`;
  }

  // lado a lado y error
  const figura = (k, extra) => `<figure><div class="marco"><img src="${I[k]}" alt="${NOMBRES[k]}"></div>
      <figcaption><strong>${NOMBRES[k]}</strong>${extra ? ' · ' + extra : ''}</figcaption></figure>`;
  const ext = (k) => !m ? '' : k === 'bicubica' ? `${num(m.psnr_bicubica)} dB` : k === 'eusr' ? `${num(m.psnr_eusr)} dB` : '';
  $('rejilla').innerHTML = ['entrada', 'bicubica', 'eusr', 'referencia'].filter((k) => k in I).map((k) => figura(k, ext(k))).join('');
  $('rejilla-error').innerHTML = validar ? figura('error_bicubica') + figura('error_eusr') : '';

  // pie
  const [W, H] = d.salida;
  $('info-resultado').textContent = (validar
    ? `Zona ${d.zona[2]}×${d.zona[3]} px reducida a ${d.zona[2] / 4}×${d.zona[3] / 4} y reconstruida`
    : `Zona ${d.zona[2]}×${d.zona[3]} px → ${W}×${H} px`) +
    ` · ${d.mosaicos.toLocaleString('es-CO')} mosaicos · ${num(d.segundos, 1)} s` +
    (d.autoensamble ? ' · autoensamble ×8' : '');
  $('descarga-eusr').href = I.eusr;
  $('descarga-bic').href = I.bicubica;

  cambiarVista();
  ponerComparacion();
  $('resultado').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function ponerComparacion() {
  const I = S.resultado.imagenes, k = $('izquierda').value;
  $('img-der').src = I.eusr;
  $('img-izq').src = I[k];
  $('rotulo-izq').textContent = NOMBRES[k].replace('Interpolación ', '').replace(/^./, (c) => c.toUpperCase());
  aplicarZoom();
  moverDivisor(S.pos);
}

function aplicarZoom() {
  const z = $('zoom').value, comp = $('comp'), img = $('img-der');
  const fijar = () => {
    if (z === 'ajustar') {
      const cajaW = $('caja-comp').clientWidth;
      const maxH = window.innerHeight * 0.7;
      const k = Math.min(cajaW / img.naturalWidth, maxH / img.naturalHeight);
      comp.style.width = `${Math.round(img.naturalWidth * Math.max(k, 0))}px`;
    } else {
      comp.style.width = `${img.naturalWidth * Number(z)}px`;
    }
    document.querySelectorAll('.rejilla img').forEach((im) => {
      im.style.width = z === 'ajustar' ? '100%' : `${im.naturalWidth * Number(z)}px`;
      im.style.maxWidth = 'none';
    });
  };
  if (img.complete && img.naturalWidth) fijar(); else img.onload = fijar;
}

function moverDivisor(p) {
  S.pos = Math.min(100, Math.max(0, p));
  $('img-izq').style.clipPath = `inset(0 ${100 - S.pos}% 0 0)`;
  $('divisor').style.left = `${S.pos}%`;
  $('divisor').setAttribute('aria-valuenow', String(Math.round(S.pos)));
}

const comp = $('comp');
let deslizando = false;
const posDesde = (e) => { const r = comp.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; };
comp.addEventListener('pointerdown', (e) => { deslizando = true; comp.setPointerCapture(e.pointerId); moverDivisor(posDesde(e)); });
comp.addEventListener('pointermove', (e) => { if (deslizando) moverDivisor(posDesde(e)); });
comp.addEventListener('pointerup', () => { deslizando = false; });
comp.addEventListener('pointercancel', () => { deslizando = false; });
$('divisor').addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') { moverDivisor(S.pos - 2); e.preventDefault(); }
  if (e.key === 'ArrowRight') { moverDivisor(S.pos + 2); e.preventDefault(); }
});

function cambiarVista() {
  const v = vista();
  $('vista-deslizador').hidden = v !== 'deslizador';
  $('vista-lado').hidden = v !== 'lado';
  $('vista-error').hidden = v !== 'error';
  aplicarZoom();
}
document.querySelectorAll('input[name="vista"]').forEach((r) => r.addEventListener('change', cambiarVista));
$('zoom').addEventListener('change', aplicarZoom);
$('izquierda').addEventListener('change', ponerComparacion);
window.addEventListener('resize', () => { if (S.resultado) aplicarZoom(); });

iniciar();
