from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"


DATASETS = {
    "matches": RAW_DATA_DIR / "matches_dataset.csv",
    "historical_players": RAW_DATA_DIR / "all_players_normalized.csv",
    "laliga_players": RAW_DATA_DIR / "players_laliga.csv",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve un DataFrame de pandas.

    Parameters
    ----------
    path : Path
        Ruta del archivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset cargado en memoria.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se ha encontrado el archivo: {path}")

    return pd.read_csv(path)


def build_dataset_summary(name: str, df: pd.DataFrame) -> str:
    """
    Genera un resumen textual básico de un dataset.

    El objetivo es documentar dimensiones, columnas, valores nulos
    y duplicados antes de construir variables predictoras.
    """
    lines = []

    lines.append(f"# Dataset: {name}")
    lines.append("")
    lines.append(f"- Filas: {df.shape[0]}")
    lines.append(f"- Columnas: {df.shape[1]}")
    lines.append(f"- Filas duplicadas: {df.duplicated().sum()}")
    lines.append("")

    lines.append("## Columnas")
    lines.append("")
    for column in df.columns:
        lines.append(f"- {column} ({df[column].dtype})")

    lines.append("")
    lines.append("## Valores nulos por columna")
    lines.append("")

    null_counts = df.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    if null_counts.empty:
        lines.append("No se han detectado valores nulos.")
    else:
        for column, count in null_counts.items():
            percentage = count / len(df) * 100
            lines.append(f"- {column}: {count} nulos ({percentage:.2f}%)")

    lines.append("")
    lines.append("## Primeras filas")
    lines.append("")
    lines.append(df.head().to_markdown(index=False))

    return "\n".join(lines)


def main() -> None:
    """
    Ejecuta el diagnóstico inicial de los tres datasets del proyecto.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    full_report = []

    for dataset_name, dataset_path in DATASETS.items():
        print(f"Cargando dataset: {dataset_name}")

        df = load_dataset(dataset_path)
        summary = build_dataset_summary(dataset_name, df)

        full_report.append(summary)
        full_report.append("\n---\n")

    output_path = REPORTS_DIR / "01_data_diagnostics.md"
    output_path.write_text("\n".join(full_report), encoding="utf-8")

    print(f"Informe generado correctamente en: {output_path}")


if __name__ == "__main__":
    main()