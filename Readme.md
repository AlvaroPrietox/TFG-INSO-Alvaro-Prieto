## Ejecución del pipeline

El proyecto incluye un script maestro de ejecución ubicado en:

```text
scripts/00_run_full_pipeline.py
```

Este script permite ejecutar el pipeline completo o fases concretas del sistema. Su objetivo es facilitar la reproducibilidad del proyecto, evitando tener que lanzar manualmente cada script de procesamiento, entrenamiento, evaluación y generación de resultados.

Además, el script genera un log automático de ejecución en:

```text
reports/00_pipeline_execution_log.md
```

Este log registra los comandos ejecutados, la salida de consola, el estado de cada paso y el tiempo de ejecución.

---

### Ejecutar pipeline completo

Ejecuta todos los pasos del proyecto en orden, incluyendo los tests al final:

```powershell
python .\scripts\00_run_full_pipeline.py
```

Esta es la opción recomendada para reproducir el proyecto completo desde cero.

---

### Ejecutar pipeline completo sin tests

Ejecuta todos los scripts del pipeline, pero omite la ejecución de `pytest` al final:

```powershell
python .\scripts\00_run_full_pipeline.py --skip-tests
```

Esta opción puede ser útil cuando se desea regenerar todos los resultados sin volver a ejecutar las pruebas unitarias.

---

### Ver fases disponibles

Muestra todas las fases que pueden ejecutarse de forma independiente:

```powershell
python .\scripts\00_run_full_pipeline.py --list-stages
```

---

### Ejecutar solo una fase concreta

El script permite ejecutar únicamente una fase específica del pipeline mediante el argumento `--stage`.

Por ejemplo, para ejecutar solo la fase del modelo final:

```powershell
python .\scripts\00_run_full_pipeline.py --stage final_model
```

Para ejecutar solo los tests:

```powershell
python .\scripts\00_run_full_pipeline.py --stage tests
```

---

### Omitir comprobación de archivos de entrada

Por defecto, cuando se ejecuta el pipeline completo, el script comprueba que existan los datasets originales necesarios en:

```text
data/raw/
```

Para omitir esa comprobación inicial:

```powershell
python .\scripts\00_run_full_pipeline.py --skip-input-check
```

Esta opción puede ser útil si se está ejecutando una fase concreta o si ya se ha comprobado manualmente que los archivos existen.

---

### Combinar opciones

También es posible combinar argumentos.

Por ejemplo, para ejecutar todo el pipeline sin tests y sin comprobar los archivos de entrada:

```powershell
python .\scripts\00_run_full_pipeline.py --skip-tests --skip-input-check
```

---

## Fases disponibles

El script maestro permite ejecutar las siguientes fases:

| Fase | Descripción |
|---|---|
| `all` | Ejecuta todo el pipeline completo |
| `diagnostics` | Realiza el diagnóstico inicial de los datasets |
| `preprocessing` | Ejecuta la limpieza y el preprocesamiento de datos |
| `baseline` | Construye y evalúa el baseline descriptivo inicial |
| `temporal_dataset` | Construye el dataset temporal jugador-temporada |
| `temporal_model` | Entrena y evalúa el modelo temporal inicial |
| `random_forest_legacy` | Entrena y genera predicciones con el modelo Random Forest inicial |
| `external_evaluation` | Ejecuta la evaluación externa con datos de LaLiga 2025-2026 |
| `summaries` | Genera resúmenes y figuras iniciales de resultados |
| `error_analysis` | Analiza el error por rangos de `xG_90` |
| `model_comparison` | Compara modelos en validación temporal y evaluación externa |
| `final_model` | Entrena el modelo final Ridge y genera sus predicciones |
| `tests` | Ejecuta las pruebas unitarias del proyecto |

---

## Uso recomendado

Para reproducir completamente el proyecto desde cero:

```powershell
python .\scripts\00_run_full_pipeline.py
```

Para comprobar únicamente que el entorno y las funciones críticas siguen funcionando:

```powershell
python .\scripts\00_run_full_pipeline.py --stage tests
```

Para regenerar únicamente el modelo final y sus predicciones:

```powershell
python .\scripts\00_run_full_pipeline.py --stage final_model
```

Para regenerar solo la comparación de modelos:

```powershell
python .\scripts\00_run_full_pipeline.py --stage model_comparison
```

Para regenerar únicamente la evaluación externa:

```powershell
python .\scripts\00_run_full_pipeline.py --stage external_evaluation
```

---

## Archivos generados por el script maestro

Durante la ejecución completa, el pipeline regenera los principales archivos procesados del proyecto:

```text
data/processed/
models/
reports/
reports/figures/
```

El archivo de log principal es:

```text
reports/00_pipeline_execution_log.md
```

Este archivo permite revisar posteriormente qué pasos se ejecutaron, cuánto tardaron y si finalizaron correctamente.

---

## Consideraciones de ejecución

Todos los comandos deben lanzarse desde la raíz del proyecto:

```text
INSO/
```

Antes de ejecutar el pipeline completo, se recomienda activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Después, se puede ejecutar el script maestro:

```powershell
python .\scripts\00_run_full_pipeline.py
```

Si algún paso falla, la ejecución se detiene automáticamente y el error queda registrado en:

```text
reports/00_pipeline_execution_log.md
```

Esto permite identificar el punto exacto del fallo y revisar la salida generada por el script correspondiente.

---

## Papel del script maestro dentro del proyecto

El script `00_run_full_pipeline.py` actúa como punto de entrada principal del sistema. Su función no es implementar nueva lógica predictiva, sino coordinar de forma ordenada los distintos módulos y scripts del proyecto.

Desde el punto de vista de Ingeniería de Software, este script mejora la reproducibilidad, la trazabilidad experimental y la facilidad de mantenimiento del sistema, ya que centraliza la ejecución de las fases principales del pipeline.

```text
datos originales → preprocesamiento → dataset temporal → modelos → evaluación → informes → tests
```

De esta forma, el proyecto puede reproducirse de manera controlada y documentada.