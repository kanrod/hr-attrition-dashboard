# IBM HR Analytics — Dashboard de Rotación Laboral

**Programación para Ciencia de Datos II**  
Juan Camilo Rodríguez Fontecha · Fundación Universitaria Compensar

---

## Descripción

Dashboard interactivo construido con **Dash + Plotly** que presenta los resultados del análisis estadístico del dataset IBM HR Analytics Employee Attrition. Incluye:

- 📊 **Resumen del Dataset** — KPIs, distribución de Attrition, histograma interactivo por departamento
- 🧪 **Contraste de Hipótesis** — Prueba T de Welch sobre MonthlyIncome
- 📈 **Regresión Lineal Múltiple** — Coeficientes β, R², RMSE, predicho vs real
- 🎯 **Regresión Logística** — Matriz de confusión, métricas con slider de umbral, simulador de riesgo

---

## ▶️ Ejecutar Localmente

### 1. Clona el repositorio
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 3. Descarga el dataset
Descarga el archivo `WA_Fn-UseC_-HR-Employee-Attrition.csv` desde:  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
y colócalo en la misma carpeta que `app.py`.

### 4. Ejecuta el dashboard
```bash
python app.py
```

Abre tu navegador en: http://127.0.0.1:8050

---

## ☁️ Ver en Binder (sin instalar nada)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TU_USUARIO/TU_REPOSITORIO/HEAD?urlpath=%2Fproxy%2F8050%2F)

> **Nota:** El botón de arriba funcionará una vez que hayas reemplazado `TU_USUARIO` y `TU_REPOSITORIO` con tus datos reales de GitHub.

---

## Estructura del Proyecto

```
hr_dashboard/
├── app.py                          # Dashboard principal (Dash)
├── requirements.txt                # Dependencias Python
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset (agregar manualmente)
├── README.md                       # Este archivo
└── links.txt                       # Links de GitHub y Binder
```

---

## Dataset

IBM HR Analytics Employee Attrition & Performance  
Fuente: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset  
1,470 registros · 35 variables · Sin valores nulos

---

## Tecnologías

- Python 3.10+
- Dash 2.17 / Plotly 5.22
- Scikit-learn 1.5
- Pandas / NumPy / SciPy
