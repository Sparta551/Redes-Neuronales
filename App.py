# -*- coding: utf-8 -*-
"""
Caso de Estudio Final - Redes Neuronales (MACC 2026-2)
Integrantes: Erwin - Laura - Camilo

Aplicación interactiva con Gradio con 3 módulos:
  1. Modelos clásicos de redes neuronales (exploración, clasificación de fallas y
     regresión del consumo energético).
  2. Redes convolucionales: clasificador de Pokémon (EfficientNetV2B0 + fine-tuning).
  3. Generación de datos sintéticos con VAE y GAN (Pokémon 64x64).

Los modelos ya están entrenados (archivos .keras en "modelos guardados"); la app solo
los carga y hace inferencia, no vuelve a entrenar.

Ejecución:
    pip install tensorflow keras gradio pandas numpy scikit-learn matplotlib seaborn pillow
    python App.py
"""

import os
import json
import time
import zipfile
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             confusion_matrix, roc_auc_score, r2_score,
                             mean_squared_error, mean_absolute_error)

import tensorflow as tf
import keras
import gradio as gr

# ============================================================================
# Rutas y configuración
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos guardados"
EXTRA_DIR = BASE_DIR / "archivos extra"

URL_DATOS = ("https://raw.githubusercontent.com/Fabian830348/Bases_Datos/refs/heads/master/"
             "Casos_Estudio/Datos/Datos_energia_Caso_Final_ML_2026.csv")
CSV_LOCAL = EXTRA_DIR / "Datos_energia_Caso_Final_ML_2026.csv"

FEATURES = ["Temperatura", "Presión", "Vibración", "Horas de operación", "Carga", "Antigüedad"]
UNIDADES = {"Temperatura": "°C", "Presión": "bar", "Vibración": "mm/s",
            "Horas de operación": "h", "Carga": "%", "Antigüedad": "años",
            "Consumo energético": "kWh", "Falla del equipo": "0 = No, 1 = Sí"}
OBJ_CLF, OBJ_REG = "Falla del equipo", "Consumo energético"

# Paleta coherente con los notebooks
MORADO, MORADO_OSC, LILA = "#7b4fc9", "#20165F", "#9386E3"
COL_REAL, COL_VAE, COL_GAN = "#2a78d6", "#eb6834", "#1baf7a"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


# ============================================================================
# MÓDULO 1 · Datos y modelos clásicos
# ============================================================================
def cargar_datos():
    """Lee el CSV (copia local si existe; si no, lo descarga y guarda una copia)."""
    if CSV_LOCAL.exists():
        crudo = pd.read_csv(CSV_LOCAL, encoding="utf-8-sig")
    else:
        crudo = pd.read_csv(URL_DATOS, encoding="utf-8-sig")
        try:
            EXTRA_DIR.mkdir(exist_ok=True)
            crudo.to_csv(CSV_LOCAL, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    crudo.columns = [c.strip() for c in crudo.columns]
    return crudo


def limpiar_datos(crudo):
    """Mismo preprocesamiento del notebook del Punto 1."""
    d = crudo.copy()
    d["Vibración"] = pd.to_numeric(d["Vibración"], errors="coerce")
    d = d.dropna(how="any").copy()
    d[OBJ_CLF] = (d[OBJ_CLF].astype(str).str.strip().str.lower()
                  .map({"sí": 1, "si": 1, "no": 0}).astype(int))
    return d


datos_crudos = cargar_datos()
datos = limpiar_datos(datos_crudos)

# ---- Reconstrucción de las particiones y escaladores (idénticos al notebook) ----
# Clasificación: 60/20/20 estratificado, random_state=0
X_all, y_clf = datos[FEATURES], datos[OBJ_CLF]
Xc_train, Xc_tmp, yc_train, yc_tmp = train_test_split(X_all, y_clf, random_state=0,
                                                      stratify=y_clf, test_size=0.4)
Xc_val, Xc_test, yc_val, yc_test = train_test_split(Xc_tmp, yc_tmp, random_state=0,
                                                    stratify=yc_tmp, test_size=0.5)
escalar_clf = StandardScaler().fit(Xc_train)

# Regresión: 60/20/20, random_state=0
y_reg = datos[OBJ_REG]
Xr_train, Xr_tmp, yr_train, yr_tmp = train_test_split(X_all, y_reg, test_size=0.4, random_state=0)
Xr_val, Xr_test, yr_val, yr_test = train_test_split(Xr_tmp, yr_tmp, test_size=0.5, random_state=0)
escalar_reg = StandardScaler().fit(Xr_train)

modelo_clf = keras.models.load_model(MODELOS_DIR / "modelo_clasificacion.keras", compile=False)
modelo_reg = keras.models.load_model(MODELOS_DIR / "modelo_regresion.keras", compile=False)


def _a_array(X):
    return np.asarray(X[FEATURES] if isinstance(X, pd.DataFrame) else X, dtype=np.float32)


def prob_falla(X):
    return modelo_clf.predict(escalar_clf.transform(_a_array(X)), verbose=0).ravel()


def pred_consumo(X):
    return modelo_reg.predict(escalar_reg.transform(_a_array(X)), verbose=0).ravel()


# ---- Métricas de clasificación (conjunto de prueba) ----
p_test = prob_falla(Xc_test)
yhat_test = (p_test > 0.5).astype(int)
cm_clf = confusion_matrix(yc_test, yhat_test)
tn, fp, fn, tp = cm_clf.ravel()


def _metricas_clf(y, p):
    yh = (p > 0.5).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y, yh).ravel()
    return [accuracy_score(y, yh), recall_score(y, yh, zero_division=0),
            tn_ / (tn_ + fp_), precision_score(y, yh, zero_division=0),
            f1_score(y, yh, zero_division=0), roc_auc_score(y, p)]


tabla_clf = pd.DataFrame({
    "Métrica": ["Accuracy", "Recall (sensibilidad)", "Specificity", "Precision", "F1-score", "AUC-ROC"],
    "Entrenamiento": _metricas_clf(yc_train, prob_falla(Xc_train)),
    "Validación": _metricas_clf(yc_val, prob_falla(Xc_val)),
    "Prueba (test)": _metricas_clf(yc_test, p_test),
}).round(4)


def fig_matriz_confusion():
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    etiquetas = np.array([[f"VN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"VP\n{tp}"]])
    sns.heatmap(cm_clf, annot=etiquetas, fmt="", cmap="Purples", cbar=False, ax=ax,
                annot_kws={"size": 16, "weight": "bold"}, linewidths=2, linecolor="white",
                xticklabels=["No falla", "Falla"], yticklabels=["No falla", "Falla"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión (test, umbral 0.5)")
    fig.tight_layout()
    return fig


def fig_dist_probabilidades():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for clase, color, nombre in [(0, LILA, "Real: No falla"), (1, MORADO_OSC, "Real: Falla")]:
        ax.hist(p_test[yc_test.values == clase], bins=25, alpha=0.7, color=color, label=nombre)
    ax.axvline(0.5, color="crimson", ls="--", label="Umbral 0.5")
    ax.set_xlabel("Probabilidad de falla predicha"); ax.set_ylabel("Equipos")
    ax.set_title("Distribución de probabilidades (test)"); ax.legend()
    fig.tight_layout()
    return fig


# ---- Métricas de regresión ----
def _metricas_reg(y, yh):
    mse = mean_squared_error(y, yh)
    return [r2_score(y, yh), mse, np.sqrt(mse), mean_absolute_error(y, yh)]


yr_pred_test = pred_consumo(Xr_test)
tabla_reg = pd.DataFrame({
    "Métrica": ["R²", "MSE", "RMSE", "MAE"],
    "Entrenamiento": _metricas_reg(yr_train, pred_consumo(Xr_train)),
    "Validación": _metricas_reg(yr_val, pred_consumo(Xr_val)),
    "Prueba (test)": _metricas_reg(yr_test, yr_pred_test),
}).round(4)


def fig_real_vs_pred():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(yr_test, yr_pred_test, s=12, alpha=0.5, color=MORADO)
    lim = [min(yr_test.min(), yr_pred_test.min()), max(yr_test.max(), yr_pred_test.max())]
    axes[0].plot(lim, lim, "--", color="crimson", label="Predicción perfecta")
    axes[0].set_xlabel("Consumo real (kWh)"); axes[0].set_ylabel("Consumo predicho (kWh)")
    axes[0].set_title("Real vs predicho (test)"); axes[0].legend()
    residuos = yr_test.values - yr_pred_test
    axes[1].hist(residuos, bins=40, color=LILA, edgecolor="white")
    axes[1].axvline(0, color="crimson", ls="--")
    axes[1].set_xlabel("Residuo (kWh)"); axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de los residuos (test)")
    fig.tight_layout()
    return fig


# ---- Exploración ----
RANGOS = {c: (float(datos[c].min()), float(datos[c].max()), float(datos[c].median())) for c in FEATURES}
VARS_NUM = FEATURES + [OBJ_REG, OBJ_CLF]


def tabla_info():
    return pd.DataFrame({
        "Variable": datos_crudos.columns,
        "Tipo original": [str(t) for t in datos_crudos.dtypes],
        "Nulos originales": [int(datos_crudos[c].isna().sum()) for c in datos_crudos.columns],
        "Nulos tras convertir a número": [
            int(pd.to_numeric(datos_crudos[c], errors="coerce").isna().sum()) if c != OBJ_CLF else 0
            for c in datos_crudos.columns],
        "Unidad": [UNIDADES.get(c, "") for c in datos_crudos.columns],
    })


def primeras_filas(n):
    return datos.head(int(n)).reset_index(drop=True)


def estadisticas():
    return datos.describe().T.round(3).reset_index(names="Variable")


def grafico_variable(variable, tipo, segmentar, bins):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hue = OBJ_CLF if (segmentar and variable != OBJ_CLF) else None
    pal = {0: LILA, 1: MORADO_OSC}
    if tipo == "Histograma":
        if variable == OBJ_CLF:
            conteo = datos[OBJ_CLF].map({0: "No", 1: "Sí"}).value_counts()
            ax.bar(conteo.index, conteo.values, color=[LILA, MORADO_OSC])
            for i, v in enumerate(conteo.values):
                ax.text(i, v, f"{v} ({v / len(datos):.1%})", ha="center", va="bottom")
        else:
            sns.histplot(data=datos, x=variable, hue=hue, bins=int(bins), kde=True,
                         palette=pal if hue else None, color=None if hue else MORADO,
                         multiple="layer" if hue else "layer", alpha=0.55, ax=ax)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Histograma de {variable}" + (" según falla" if hue else ""))
    else:
        if hue:
            d = datos.assign(Falla=datos[OBJ_CLF].map({0: "No", 1: "Sí"}))
            sns.boxplot(data=d, x="Falla", y=variable, hue="Falla",
                        palette={"No": LILA, "Sí": MORADO_OSC}, legend=False, ax=ax)
        else:
            sns.boxplot(data=datos, y=variable, color=LILA, ax=ax)
        ax.set_title(f"Boxplot de {variable}" + (" según falla" if hue else ""))
    unidad = UNIDADES.get(variable, "")
    if tipo == "Histograma":
        ax.set_xlabel(f"{variable} ({unidad})" if unidad else variable)
    else:
        ax.set_ylabel(f"{variable} ({unidad})" if unidad else variable)
    fig.tight_layout()
    return fig


def grafico_dispersion(x, y):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    d = datos.assign(Falla=datos[OBJ_CLF].map({0: "No", 1: "Sí"}))
    sns.scatterplot(data=d, x=x, y=y, hue="Falla", palette={"No": LILA, "Sí": MORADO_OSC},
                    alpha=0.6, s=18, ax=ax)
    r = datos[x].corr(datos[y])
    ax.set_title(f"{x} vs {y}  (correlación de Pearson = {r:.2f})")
    fig.tight_layout()
    return fig


def grafico_correlaciones():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(datos[VARS_NUM].corr(), annot=True, fmt=".2f", cmap="Purples",
                vmin=-1, vmax=1, square=True, ax=ax, annot_kws={"size": 9})
    ax.set_title("Mapa de correlaciones")
    fig.tight_layout()
    return fig


# ---- Predicción con nuevos registros ----
def _fila(*vals):
    return pd.DataFrame([vals], columns=FEATURES).astype(float)


def predecir_falla(*vals):
    p = float(prob_falla(_fila(*vals))[0])
    clase = "Sí" if p > 0.5 else "No"
    if p > 0.8:
        riesgo = "🔴 **Riesgo alto**: priorizar inspección y mantenimiento."
    elif p > 0.5:
        riesgo = "🟠 **Riesgo moderado-alto**: programar revisión."
    elif p > 0.3:
        riesgo = "🟡 **Riesgo moderado**: monitorear las condiciones de operación."
    else:
        riesgo = "🟢 **Riesgo bajo**: operación normal."
    md = f"### ¿Falla del equipo? **{clase}**\nProbabilidad estimada de falla: **{p:.1%}**\n\n{riesgo}"
    return {"Falla (Sí)": p, "Sin falla (No)": 1 - p}, md


def predecir_consumo(*vals):
    c = float(pred_consumo(_fila(*vals))[0])
    mae = tabla_reg.loc[tabla_reg["Métrica"] == "MAE", "Prueba (test)"].item()
    media = datos[OBJ_REG].mean()
    comp = "por encima" if c > media else "por debajo"
    return (f"### Consumo energético estimado: **{c:,.2f} kWh**\n"
            f"Error típico del modelo (MAE en test): ± {mae:.2f} kWh  \n"
            f"Está {comp} del consumo promedio de la planta ({media:,.2f} kWh).")


def _validar_lote(df):
    if df is None or len(df) == 0:
        raise gr.Error("Ingrese al menos un registro en la tabla.")
    df = pd.DataFrame(df).copy()
    df.columns = FEATURES[:len(df.columns)] if len(df.columns) == len(FEATURES) else df.columns
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        raise gr.Error(f"Faltan columnas: {faltan}")
    df = df[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise gr.Error("Los registros deben tener valores numéricos en todas las columnas.")
    return df


def predecir_falla_lote(df):
    df = _validar_lote(df)
    p = prob_falla(df)
    out = df.copy()
    out["Prob. de falla"] = np.round(p.astype(float), 4)
    out["Predicción"] = np.where(p > 0.5, "Sí", "No")
    return out


def predecir_consumo_lote(df):
    df = _validar_lote(df)
    out = df.copy()
    out["Consumo predicho (kWh)"] = np.round(pred_consumo(df).astype(float), 2)
    return out


def ejemplo_clf_tabla():
    m = Xc_test.sample(5, random_state=7)
    out = m.copy().reset_index(drop=True)
    return out, yc_test.loc[m.index].map({0: "No", 1: "Sí"}).tolist()


# ============================================================================
# MÓDULO 2 · CNN (Pokémon)
# ============================================================================
with open(EXTRA_DIR / "clases_pokemon.json", encoding="utf-8") as f:
    META_CNN = json.load(f)
CLASES_CNN = META_CNN["clases"]
IMG_SIZE_CNN = int(META_CNN.get("img_size", 224))
try:
    METRICAS_CNN = pd.read_csv(EXTRA_DIR / "metricas_cnn.csv")
    for c in METRICAS_CNN.columns[1:]:
        METRICAS_CNN[c] = (METRICAS_CNN[c] * 100).round(2).astype(str) + " %"
except Exception:
    METRICAS_CNN = pd.DataFrame()

_modelo_cnn = None


def get_modelo_cnn():
    """Carga diferida: el modelo CNN pesa ~70 MB, se carga en la primera predicción."""
    global _modelo_cnn
    if _modelo_cnn is None:
        _modelo_cnn = keras.models.load_model(MODELOS_DIR / "modelo_cnn_pokemon.keras", compile=False)
    return _modelo_cnn


def preparar_imagen(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB").resize((IMG_SIZE_CNN, IMG_SIZE_CNN), Image.BILINEAR)
    # EfficientNetV2 incluye su propio preprocesamiento: recibe valores 0-255
    return np.asarray(img, dtype=np.float32)[None]


def clasificar_imagen(img, top_k):
    if img is None:
        raise gr.Error("Primero cargue una imagen.")
    p = get_modelo_cnn().predict(preparar_imagen(img), verbose=0)[0].astype(float)
    top = np.argsort(p)[::-1][:int(top_k)]
    res = {CLASES_CNN[i]: float(p[i]) for i in top}
    nombre, prob = CLASES_CNN[top[0]], p[top[0]]
    if prob >= 0.6:
        conf = "El modelo está **bastante seguro** de la predicción."
    elif prob >= 0.3:
        conf = "Confianza **media**: revise también las otras clases del top."
    else:
        conf = "Confianza **baja**: la imagen podría ser ambigua o de un Pokémon fuera de las generaciones I–III."
    md = f"## {nombre}\nProbabilidad estimada: **{prob:.1%}**\n\n{conf}"

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(top) + 1.2))
    nombres = [CLASES_CNN[i] for i in top][::-1]
    vals = [p[i] for i in top][::-1]
    ax.barh(nombres, vals, color=[LILA] * (len(vals) - 1) + [MORADO])
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    ax.set_xlim(0, 1.1); ax.set_xlabel("Probabilidad")
    ax.set_title(f"Top-{len(top)} clases más probables")
    fig.tight_layout()
    return md, res, fig


# ============================================================================
# MÓDULO 3 · Generación sintética (VAE / GAN)
# ============================================================================
LATENT_VAE, LATENT_GAN, IMG_SIZE_GEN = 128, 128, 64
_gen = {}


def get_generador(nombre):
    if nombre not in _gen:
        archivo = "vae_decoder.keras" if nombre == "VAE" else "gan_generador.keras"
        _gen[nombre] = keras.models.load_model(MODELOS_DIR / archivo, compile=False)
    return _gen[nombre]


def generar_imagenes(modelo, n, semilla=None):
    rng = np.random.default_rng(None if semilla in (None, "", -1) else int(semilla))
    m = get_generador(modelo)
    if modelo == "VAE":
        z = rng.normal(size=(n, LATENT_VAE)).astype("float32")
        return np.clip(m.predict(z, verbose=0, batch_size=256), 0, 1)       # sigmoid -> [0,1]
    z = rng.normal(size=(n, LATENT_GAN)).astype("float32")
    return np.clip((m.predict(z, verbose=0, batch_size=256) + 1) / 2, 0, 1)  # tanh -> [0,1]


def caracteristicas(imgs, lote=256):
    """Las mismas 9 características interpretables del notebook del Punto 3."""
    filas = []
    for i in range(0, len(imgs), lote):
        b = tf.convert_to_tensor(np.clip(imgs[i:i + lote], 0, 1), tf.float32)
        gris = tf.image.rgb_to_grayscale(b)[..., 0]
        hsv = tf.image.rgb_to_hsv(b)
        fondo = tf.reduce_all(b > 0.9, axis=-1)
        objeto = tf.cast(~fondo, tf.float32)
        sat = tf.reduce_sum(hsv[..., 1] * objeto, [1, 2]) / (tf.reduce_sum(objeto, [1, 2]) + 1e-6)
        R, G, B = b[..., 0], b[..., 1], b[..., 2]
        rg, yb = R - G, 0.5 * (R + G) - B
        colorido = (tf.sqrt(tf.math.reduce_std(rg, [1, 2]) ** 2 + tf.math.reduce_std(yb, [1, 2]) ** 2)
                    + 0.3 * tf.sqrt(tf.reduce_mean(rg, [1, 2]) ** 2 + tf.reduce_mean(yb, [1, 2]) ** 2))
        sob = tf.image.sobel_edges(gris[..., None])
        nitidez = tf.reduce_mean(tf.sqrt(tf.reduce_sum(sob ** 2, -1) + 1e-12), [1, 2, 3])
        filas.append(np.stack([
            tf.reduce_mean(gris, [1, 2]), tf.math.reduce_std(gris, [1, 2]),
            tf.reduce_mean(R, [1, 2]), tf.reduce_mean(G, [1, 2]), tf.reduce_mean(B, [1, 2]),
            sat, colorido, tf.reduce_mean(tf.cast(fondo, tf.float32), [1, 2]), nitidez], axis=1))
    cols = ["brillo", "contraste", "media_R", "media_G", "media_B",
            "saturacion", "colorido", "frac_fondo", "nitidez"]
    return pd.DataFrame(np.concatenate(filas), columns=cols)


def _a_pil(im, escala=3):
    pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
    return pil.resize((IMG_SIZE_GEN * escala, IMG_SIZE_GEN * escala), Image.NEAREST)


def fig_resumen_sinteticos(feats, modelo):
    color = COL_VAE if modelo == "VAE" else COL_GAN
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, col in zip(axes, ["brillo", "saturacion", "frac_fondo", "nitidez"]):
        ax.hist(feats[col], bins=25, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(col); ax.set_ylabel("Frecuencia")
    fig.suptitle(f"Distribución de características de los datos generados con {modelo}",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def generar_ui(modelo, n, semilla):
    t0 = time.time()
    imgs = generar_imagenes(modelo, int(n), semilla)
    feats = caracteristicas(imgs)
    galeria = [(_a_pil(im), f"{modelo} #{i + 1}") for i, im in enumerate(imgs[:10])]
    tabla = feats.head(10).round(3).reset_index(names="id")
    tabla["id"] = tabla["id"] + 1
    info = (f"Se generaron **{len(imgs)} registros** (imágenes de 64×64×3) con el modelo "
            f"**{modelo}** en {time.time() - t0:.1f} s. Abajo se visualizan 10 de ellos.")
    return galeria, tabla, fig_resumen_sinteticos(feats, modelo), {"modelo": modelo, "imgs": imgs}, info, None


def exportar_ui(estado, escala):
    if not estado or estado.get("imgs") is None:
        raise gr.Error("Primero genere los datos con el botón «Generar».")
    imgs, modelo = estado["imgs"], estado["modelo"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    ruta_zip = Path(tempfile.gettempdir()) / f"sinteticos_{modelo}_{len(imgs)}_{ts}.zip"
    feats = caracteristicas(imgs)
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, im in enumerate(imgs):
            buf = tempfile.SpooledTemporaryFile()
            _a_pil(im, int(escala)).save(buf, format="PNG")
            buf.seek(0)
            zf.writestr(f"imagenes/{modelo}_{i + 1:04d}.png", buf.read())
        zf.writestr(f"caracteristicas_{modelo}.csv", feats.to_csv(index_label="id"))
        # Datos crudos: cada fila es una imagen aplanada (64*64*3 valores en [0,1])
        np_buf = tempfile.SpooledTemporaryFile()
        np.save(np_buf, imgs.astype(np.float32))
        np_buf.seek(0)
        zf.writestr(f"datos_{modelo}.npy", np_buf.read())
    return str(ruta_zip)


# ============================================================================
# Interfaz
# ============================================================================
CSS = """
#encabezado {background: linear-gradient(90deg, #20165F 0%, #7b4fc9 100%);
             border-radius: 14px; padding: 18px 26px; color: white;}
#encabezado h1, #encabezado p {color: white !important; margin: 0;}
.conclusion {border-left: 5px solid #7b4fc9; padding-left: 12px !important;}
"""
TEMA = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")


def _entradas_features():
    comps = []
    for c in FEATURES:
        lo, hi, med = RANGOS[c]
        paso = 1.0 if c in ("Antigüedad",) else (10.0 if c == "Horas de operación" else 0.01)
        comps.append(gr.Slider(minimum=round(lo, 2), maximum=round(hi, 2), value=round(med, 2),
                               step=paso, label=f"{c} ({UNIDADES[c]})"))
    return comps


_major = int(gr.__version__.split(".")[0])
_blocks_kw = dict(title="Caso Final · Redes Neuronales")
if _major < 6:
    _blocks_kw.update(theme=TEMA, css=CSS)

with gr.Blocks(**_blocks_kw) as demo:
    gr.HTML("""
    <div id="encabezado">
      <h1>Caso de Estudio Final · Redes Neuronales</h1>
      <p>Pregrado MACC 2026-2 · Prof. Fabián Sánchez · Erwin · Laura · Camilo</p>
    </div>""")

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        with gr.Tab("1 · Modelos clásicos de redes neuronales"):
            gr.Markdown(
                "**Caso:** desempeño energético y fallas en equipos industriales. "
                f"{len(datos_crudos):,} registros originales → {len(datos):,} tras limpieza "
                "(se convirtió *Vibración* a numérica y se eliminaron filas con nulos).")
            with gr.Tabs():
                # ---------- Submódulo 1: Exploración ----------
                with gr.Tab("Submódulo 1 · Exploración"):
                    with gr.Row():
                        n_filas = gr.Slider(5, 50, value=10, step=1, label="Número de filas a mostrar")
                    tabla_head = gr.Dataframe(value=primeras_filas(10), label="Primeras filas (datos limpios)",
                                              interactive=False)
                    n_filas.change(primeras_filas, n_filas, tabla_head)
                    with gr.Accordion("Estructura de los datos (tipos y valores nulos)", open=False):
                        gr.Dataframe(value=tabla_info(), interactive=False)
                    gr.Dataframe(value=estadisticas(), label="Estadísticas descriptivas", interactive=False)

                    gr.Markdown("### Gráficos por variable")
                    with gr.Row():
                        with gr.Column(scale=1):
                            var_sel = gr.Dropdown(VARS_NUM, value="Consumo energético", label="Variable")
                            tipo_sel = gr.Radio(["Histograma", "Boxplot"], value="Histograma",
                                                label="Tipo de gráfico")
                            seg_sel = gr.Checkbox(value=False, label="Separar por Falla del equipo")
                            bins_sel = gr.Slider(5, 80, value=30, step=1, label="Número de bins (histograma)")
                            btn_graf = gr.Button("Graficar", variant="primary")
                        with gr.Column(scale=3):
                            plot_var = gr.Plot(value=grafico_variable("Consumo energético", "Histograma", False, 30))
                    entradas_g = [var_sel, tipo_sel, seg_sel, bins_sel]
                    btn_graf.click(grafico_variable, entradas_g, plot_var)
                    for comp in entradas_g:
                        comp.change(grafico_variable, entradas_g, plot_var)

                    gr.Markdown("### Relaciones entre variables")
                    with gr.Row():
                        with gr.Column(scale=1):
                            x_sel = gr.Dropdown(FEATURES + [OBJ_REG], value="Carga", label="Eje X")
                            y_sel = gr.Dropdown(FEATURES + [OBJ_REG], value="Consumo energético", label="Eje Y")
                        with gr.Column(scale=3):
                            plot_disp = gr.Plot(value=grafico_dispersion("Carga", "Consumo energético"))
                    x_sel.change(grafico_dispersion, [x_sel, y_sel], plot_disp)
                    y_sel.change(grafico_dispersion, [x_sel, y_sel], plot_disp)
                    with gr.Accordion("Mapa de correlaciones", open=False):
                        gr.Plot(value=grafico_correlaciones())

                # ---------- Submódulo 2: Clasificación ----------
                with gr.Tab("Submódulo 2 · Clasificación de fallas"):
                    gr.Markdown(
                        "Red neuronal densa **6 → 16 (ReLU) → 8 (ReLU) → 1 (sigmoid)**, pérdida *binary "
                        "crossentropy*, optimizador Adam, 100 épocas. División 60 % / 20 % / 20 % estratificada y "
                        "variables estandarizadas.")
                    with gr.Row():
                        with gr.Column():
                            gr.Dataframe(value=tabla_clf, label="Métricas del modelo", interactive=False)
                            gr.Plot(value=fig_matriz_confusion())
                        with gr.Column():
                            gr.Plot(value=fig_dist_probabilidades())
                            gr.Markdown(
                                f"""En el conjunto de prueba ({len(yc_test)} equipos):
- **{tp}** fallas detectadas correctamente y **{tn}** equipos sanos reconocidos.
- **{fn}** fallas no detectadas (falsos negativos) y **{fp}** falsas alarmas.""")

                    gr.Markdown("### Ingresar un nuevo registro")
                    with gr.Row():
                        with gr.Column(scale=2):
                            ent_clf = _entradas_features()
                            btn_clf = gr.Button("Predecir falla", variant="primary")
                        with gr.Column(scale=2):
                            md_clf = gr.Markdown()
                            lbl_clf = gr.Label(label="Probabilidades", num_top_classes=2)
                    btn_clf.click(predecir_falla, ent_clf, [lbl_clf, md_clf])

                    gr.Markdown("### Predecir varios registros a la vez\n"
                                "Edite la tabla (puede agregar filas) y presione **Predecir lote**. "
                                "Se cargan como ejemplo 5 equipos del conjunto de prueba.")
                    ej_clf, reales_clf = ejemplo_clf_tabla()
                    tabla_in_clf = gr.Dataframe(value=ej_clf, headers=FEATURES, interactive=True,
                                                label="Registros nuevos")
                    gr.Markdown(f"*Valor real de los 5 ejemplos cargados:* {', '.join(reales_clf)}")
                    btn_lote_clf = gr.Button("Predecir lote", variant="primary")
                    tabla_out_clf = gr.Dataframe(label="Predicciones", interactive=False)
                    btn_lote_clf.click(predecir_falla_lote, tabla_in_clf, tabla_out_clf)

                # ---------- Submódulo 3: Regresión ----------
                with gr.Tab("Submódulo 3 · Regresión del consumo"):
                    gr.Markdown(
                        "Red neuronal densa **6 → 16 (ReLU) → 1 (lineal)**, pérdida MSE, optimizador Adam. "
                        "División 60 % / 20 % / 20 % y variables estandarizadas.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Dataframe(value=tabla_reg, label="Métricas del modelo", interactive=False)
                        with gr.Column(scale=2):
                            gr.Plot(value=fig_real_vs_pred())

                    gr.Markdown("### Ingresar un nuevo registro")
                    with gr.Row():
                        with gr.Column(scale=2):
                            ent_reg = _entradas_features()
                            btn_reg = gr.Button("Predecir consumo", variant="primary")
                        with gr.Column(scale=2):
                            md_reg = gr.Markdown()
                    btn_reg.click(predecir_consumo, ent_reg, md_reg)

                    gr.Markdown("### Predecir varios registros a la vez")
                    m_reg = Xr_test.sample(5, random_state=7)
                    tabla_in_reg = gr.Dataframe(value=m_reg.reset_index(drop=True), headers=FEATURES,
                                                interactive=True, label="Registros nuevos")
                    gr.Markdown("*Consumo real de los 5 ejemplos cargados:* "
                                + ", ".join(f"{v:.2f} kWh" for v in yr_test.loc[m_reg.index]))
                    btn_lote_reg = gr.Button("Predecir lote", variant="primary")
                    tabla_out_reg = gr.Dataframe(label="Predicciones", interactive=False)
                    btn_lote_reg.click(predecir_consumo_lote, tabla_in_reg, tabla_out_reg)

            with gr.Accordion("Conclusiones del Módulo 1", open=False):
                acc = tabla_clf.loc[0, "Prueba (test)"]
                rec = tabla_clf.loc[1, "Prueba (test)"]
                spe = tabla_clf.loc[2, "Prueba (test)"]
                r2 = tabla_reg.loc[0, "Prueba (test)"]
                mae = tabla_reg.loc[3, "Prueba (test)"]
                gr.Markdown(f"""
- **Exploración:** la *Carga* es la variable más relacionada con el *Consumo energético* (correlación ≈ {datos['Carga'].corr(datos[OBJ_REG]):.2f}). Las clases de falla están desbalanceadas (≈ {datos[OBJ_CLF].mean():.0%} de equipos con falla).
- **Clasificación:** accuracy de **{acc:.1%}** en test con curvas de entrenamiento y validación muy cercanas (sin sobreajuste). La especificidad (**{spe:.1%}**) es mayor que el recall (**{rec:.1%}**): el modelo reconoce muy bien los equipos sanos, pero se le escapan algunas fallas. Para mantenimiento preventivo se podría bajar el umbral de 0.5 para detectar más fallas a costa de más falsas alarmas.
- **Regresión:** el modelo explica el **{r2:.2%}** de la variabilidad del consumo (R²) con un error medio de **{mae:.2f} kWh**, muy pequeño frente a un consumo promedio de {datos[OBJ_REG].mean():.0f} kWh. Esto indica una relación casi determinística entre las condiciones de operación y el consumo.
""", elem_classes="conclusion")

        # ------------------------------------------------------------------ #
        with gr.Tab("2 · Redes convolucionales"):
            gr.Markdown(
                "### ¿Qué Pokémon es?\n"
                "Clasificador de **386 Pokémon** (generaciones I–III) entrenado con ~19.000 imágenes "
                "obtenidas por *web scraping*. Modelo: **EfficientNetV2-B0** preentrenada en ImageNet + "
                "*transfer learning*, *fine-tuning* y *data augmentation*. Cargue una imagen y presione "
                "**Clasificar**.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Imagen a clasificar", height=340)
                    topk = gr.Slider(1, 10, value=5, step=1, label="Número de clases a mostrar (top-k)")
                    btn_cnn = gr.Button("Clasificar", variant="primary")
                with gr.Column(scale=1):
                    md_cnn = gr.Markdown()
                    lbl_cnn = gr.Label(label="Probabilidades", num_top_classes=10)
            plot_cnn = gr.Plot(label="Probabilidades del top-k")
            btn_cnn.click(clasificar_imagen, [img_in, topk], [md_cnn, lbl_cnn, plot_cnn])
            img_in.upload(clasificar_imagen, [img_in, topk], [md_cnn, lbl_cnn, plot_cnn])

            with gr.Accordion("Métricas de los modelos en test y conclusiones", open=False):
                if not METRICAS_CNN.empty:
                    gr.Dataframe(value=METRICAS_CNN, interactive=False)
                gr.Markdown("""
- La **CNN entrenada desde cero** prácticamente no aprende: con ~35 imágenes de entrenamiento por clase y 386 clases no hay datos suficientes para aprender filtros útiles.
- El **transfer learning con EfficientNetV2-B0 + fine-tuning** multiplica el desempeño: aprovecha las características visuales aprendidas en ImageNet y alcanza ≈ 60 % de accuracy y ≈ 80 % de *top-5* (la clase correcta está entre las 5 más probables), frente al 0,26 % de acertar al azar.
- Los errores más frecuentes ocurren entre Pokémon muy parecidos (Plusle/Minun, Porygon/Porygon2, evoluciones de la misma familia) y por el ruido propio de un dataset descargado de internet.
""", elem_classes="conclusion")

        # ------------------------------------------------------------------ #
        with gr.Tab("3 · Generación de datos sintéticos"):
            gr.Markdown(
                "### Pokémon sintéticos con VAE y GAN\n"
                "Modelos entrenados con el *Pokemon Images Dataset* de Kaggle (819 sprites, reducidos a 64×64). "
                "Seleccione el modelo, la cantidad de registros y presione **Generar**; luego puede "
                "**Exportar datos** (ZIP con las imágenes PNG, un CSV con sus características y un `.npy` "
                "con los datos crudos).")
            estado_gen = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    modelo_gen = gr.Radio(["VAE", "GAN"], value="GAN", label="Tipo de modelo generativo")
                    n_gen = gr.Slider(10, 2000, value=100, step=10, label="Cantidad de registros a generar")
                    semilla_gen = gr.Number(value=42, precision=0, label="Semilla (-1 = aleatoria)")
                    btn_gen = gr.Button("Generar", variant="primary")
                    escala_exp = gr.Slider(1, 8, value=4, step=1, label="Escala de las imágenes exportadas (×64 px)")
                    btn_exp = gr.Button("Exportar datos", variant="secondary")
                    archivo_exp = gr.File(label="Archivo exportado")
                with gr.Column(scale=3):
                    info_gen = gr.Markdown()
                    galeria = gr.Gallery(label="10 registros generados", columns=5, rows=2,
                                         height="auto", object_fit="contain")
                    tabla_gen = gr.Dataframe(label="Características de los 10 registros mostrados",
                                             interactive=False)
                    plot_gen = gr.Plot()
            btn_gen.click(generar_ui, [modelo_gen, n_gen, semilla_gen],
                          [galeria, tabla_gen, plot_gen, estado_gen, info_gen, archivo_exp])
            btn_exp.click(exportar_ui, [estado_gen, escala_exp], archivo_exp)

            with gr.Accordion("Diferencias entre VAE y GAN y conclusiones", open=False):
                gr.Markdown("""
| Aspecto | VAE | GAN |
|---|---|---|
| Principio | Maximiza la ELBO: reconstrucción + regularización KL | Juego min-max entre generador y discriminador |
| Pérdida | Explícita, baja de forma estable | Oscila; no mide directamente la calidad |
| Nitidez | Imágenes **borrosas** (promedia las salidas posibles) | Imágenes **más nítidas** y con más contraste |
| Diversidad | Cubre bien la distribución, sin colapso de modos | Riesgo de colapso de modos |
| Espacio latente | Tiene codificador: permite reconstruir e interpolar | Solo genera a partir de ruido |

- Ambos modelos conservan la estructura general del dataset: una figura centrada sobre fondo blanco y la fuerte correlación entre los canales R, G y B.
- El **VAE** concentra sus características alrededor de la media (menos contraste, saturación y nitidez), mientras que la **GAN** reproduce mejor la textura y las colas de las distribuciones de color.
- Con solo 819 imágenes, ninguno genera Pokémon perfectamente reconocibles, pero las imágenes sintéticas son nuevas y no copias del entrenamiento.
""", elem_classes="conclusion")

    gr.Markdown("<center><sub>Los modelos se cargan ya entrenados desde la carpeta "
                "<code>modelos guardados</code>; la aplicación no vuelve a entrenarlos.</sub></center>")


if __name__ == "__main__":
    if _major >= 6:
        demo.launch(theme=TEMA, css=CSS)
    else:
        demo.launch()
