# EUSR-GAN · prototipo web

Aplicación web para probar EUSR-GAN con imágenes de ultrasonido. El usuario carga una imagen, marca con el ratón la zona que quiere mejorar y compara el resultado con la interpolación bicúbica mediante un deslizador, en vista lado a lado o, en modo validación, con mapas de error.

> **Prototipo de investigación.** No es un dispositivo médico ni está validado para diagnóstico. No use sus resultados para tomar decisiones clínicas.

## Qué hace

| Modo | Qué ocurre | Para qué sirve |
|---|---|---|
| **Ampliar zona ×4** | La zona marcada se toma como entrada y se amplía ×4 con EUSR-GAN y con interpolación bicúbica. | Mostrar el uso imaginado por el médico: ampliar una región de interés. **Exploratorio:** el modelo se entrenó para reconstruir imágenes reducidas ×4, no para ampliar la imagen adquirida, y aquí no hay referencia para medir la mejora. |
| **Validar con referencia** | La zona se reduce ×4 (bicúbica, como en el entrenamiento) y se reconstruye. La zona original sirve de referencia: se calculan PSNR y SSIM y se muestran los mapas de error. | Demostrar la mejora medible, en el mismo escenario evaluado en el artículo. |

- **Modelos:** Mama (BUSI) y Corazón (CAMUS). Se ofrecen los que tengan pesos en `pesos/`.
- **Formatos:** PNG, JPG, BMP, TIFF y DICOM. De un DICOM solo se leen los píxeles; si es un cine, se usa el primer cuadro.
- **Inferencia:** idéntica a la del artículo (mosaicos de 32×32 con paso 16 y ventana de Hann), con autoensamble ×8 opcional.
- **Privacidad:** las imágenes se procesan en memoria y no se guardan; el registro del servidor anota solo tamaños y tiempos.

## Estructura

```
eusr_web/
├── app/
│   ├── main.py          API (FastAPI): /api/estado, /api/vista, /api/mejorar
│   ├── modelo.py        arquitectura EUSR-GAN (copia literal del entrenamiento) e inferencia
│   ├── imagenes.py      lectura de PNG/JPG/TIFF/DICOM y codificación de resultados
│   └── static/          interfaz web (HTML, CSS y JavaScript, sin dependencias externas)
├── pesos/               aquí van los .weights.h5 (ver pesos/LEEME.txt)
├── requirements.txt     TensorFlow 2.12 y NumPy 1.23.5, como en el entrenamiento
├── Dockerfile
├── docker-compose.yml   servicio web y, opcionalmente, HTTPS con Caddy
├── Caddyfile
└── .env.ejemplo         usuario, clave, límites y dominio
```

## 1. Preparar los pesos

Copie desde JupyterLab (`/home/datascience`) los archivos indicados en `pesos/LEEME.txt`:

| Modelo | Archivo de origen | Destino |
|---|---|---|
| Mama (BUSI) | `resultados_busi_native_scratch_yolo/g_busi_stage1_best.weights.h5` | `pesos/eusr_busi.weights.h5` |
| Corazón (CAMUS) | `resultados_camus_native/g_camus_stage1_best.weights.h5` (o `..._stage2_best...` si `stage2_used` es `true`) | `pesos/eusr_camus.weights.h5` |

## 2. Probar en una máquina con Python 3.11

```bash
cd eusr_web
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Abra `http://localhost:8000`. En el registro deben aparecer los modelos cargados y la velocidad medida en mosaicos por segundo.

> La sesión de notebook de OCI Data Science (JupyterLab) no publica puertos hacia Internet. Para que otras personas accedan, despliegue en una instancia de cómputo, como se explica a continuación.

## 3. Desplegar en Oracle Cloud (instancia de cómputo)

**a. Crear la instancia.** Ubuntu 22.04. Sirven dos formas:

- **VM.Standard.E4.Flex** (x86) con 2 OCPU y 16 GB.
- **VM.Standard.A1.Flex** (Ampere, ARM), que entra en la capa gratuita con hasta 4 OCPU y 24 GB.

`requirements.txt` instala la versión de TensorFlow adecuada para cada arquitectura. Cada proceso usa alrededor de 1 a 2 GB de memoria.

**b. Abrir el puerto.**

1. En la consola de OCI, vaya a *Networking → Virtual Cloud Networks → (su VCN) → Security Lists* y agregue una regla de entrada TCP para el puerto 8000. Si va a usar HTTPS con dominio, abra en cambio los puertos 80 y 443. Si puede, restrinja el origen a las IP de su universidad u hospital.
2. Las imágenes Ubuntu de OCI traen además reglas de `iptables`, así que abra el puerto también en la instancia:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT   # o 80 y 443
sudo netfilter-persistent save
```

**c. Instalar Docker.**

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
```

**d. Copiar el proyecto y los pesos.** Hay dos caminos:

- Desde su computador: `scp -r eusr_web ubuntu@IP:~/`.
- Desde el notebook de OCI, a través de Object Storage:

```bash
oci os object put -bn <bucket> --file g_busi_stage1_best.weights.h5   # en JupyterLab
oci os object get -bn <bucket> --name g_busi_stage1_best.weights.h5 --file pesos/eusr_busi.weights.h5   # en la instancia
```

**e. Configurar y arrancar.**

```bash
cd ~/eusr_web
cp .env.ejemplo .env && nano .env          # defina APP_USUARIO y APP_CLAVE
docker compose up -d --build                # la primera construcción tarda unos minutos
docker compose logs -f eusr                 # verifique "Modelo 'mama' cargado..."
```

Abra `http://IP_PUBLICA:8000`.

**f. HTTPS (recomendado si se usarán imágenes reales).**

1. Apunte un dominio a la IP pública.
2. Escriba `DOMINIO=su.dominio` en `.env` y ejecute:

```bash
docker compose --profile https up -d --build
```

Caddy obtiene el certificado automáticamente. Cierre entonces el puerto 8000 en la Security List y deje solo 80 y 443.

**Actualizar los pesos:** reemplace los archivos en `pesos/` y ejecute `docker compose restart eusr`.

## 4. Uso

1. Cargue una imagen (o arrástrela al recuadro).
2. Elija el modelo y el modo.
3. Arrastre sobre la imagen para marcar la zona; arrastre dentro de ella para moverla. Use **Imagen completa** o **Borrar zona** según necesite (Esc también borra la zona).
4. Pulse **Ampliar zona ×4** o **Reconstruir y medir**. La interfaz estima el tiempo según la velocidad medida en el servidor.
5. Revise el resultado:
   - **Comparar:** deslizador que compara EUSR-GAN con la bicúbica, la entrada o la referencia.
   - **Lado a lado:** las imágenes en paralelo.
   - **Error:** solo en modo validación.

   El zoom permite verlas al 100 % o al 200 %, y se puede descargar el resultado en PNG.

## Configuración

| Variable | Por defecto | Descripción |
|---|---|---|
| `APP_USUARIO`, `APP_CLAVE` | vacías | Si se definen, la aplicación pide usuario y clave (autenticación básica). Úsela siempre junto con HTTPS. |
| `MODELO_MAMA`, `MODELO_CORAZON` | `pesos/eusr_busi.weights.h5`, `pesos/eusr_camus.weights.h5` | Rutas de los pesos. |
| `MAX_LADO_AMPLIAR` | 512 | Lado máximo de la zona en modo ampliar; la salida mide hasta 2048×2048. |
| `MAX_MB` | 25 | Tamaño máximo del archivo. |
| `LOTE` | 16 | Mosaicos por lote. Bájelo si el servidor tiene poca memoria. |

## Rendimiento orientativo (CPU, sin GPU)

El tiempo crece con el número de mosaicos de 32×32 de la entrada.

| Caso | Mosaicos | Tiempo a ~24 mosaicos/s |
|---|---|---|
| Zona de 160×120 px, modo ampliar | 63 | unos 3 s |
| Zona de 512×512 px, modo ampliar | 961 | unos 40 s |
| Validación de una imagen completa de 480×600 px | 63 | unos 3 s |
| Cualquier caso con autoensamble | ×8 | ×8 |

## Limitaciones y responsabilidad

- Es un prototipo de investigación: el modelo se entrenó con bases públicas (BUSI, CAMUS) y con degradación bicúbica. Ante imágenes de otros equipos o con otro procesamiento puede comportarse distinto, y en modo ampliar puede generar textura que no existe en el tejido.
- Para uso clínico se requerirían validación con especialistas y el trámite regulatorio correspondiente (en Colombia, ante el INVIMA).
- Las imágenes de pacientes son datos personales sensibles (Ley 1581 de 2012). Use imágenes anonimizadas, HTTPS y usuario y clave. La aplicación no guarda las imágenes, pero sí viajan por la red hasta el servidor.
