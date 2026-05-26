import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

LOG_OUTPUT_PATH = REPORTS_DIR / "00_pipeline_execution_log.md"


REQUIRED_RAW_FILES = [
    DATA_RAW_DIR / "matches_dataset.csv",
    DATA_RAW_DIR / "all_players_normalized.csv",
    DATA_RAW_DIR / "players_laliga.csv",
    DATA_RAW_DIR / "laliga_players_2526.csv",
]


@dataclass(frozen=True)
class PipelineStep:
    """
    Representa un paso ejecutable del pipeline.
    """
    step_id: str
    stage: str
    description: str
    command: list[str]


def python_script(script_name: str) -> list[str]:
    """
    Construye el comando para ejecutar un script Python usando el intérprete
    activo, normalmente el del entorno virtual.
    """
    return [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
    ]


PIPELINE_STEPS = [
    PipelineStep(
        step_id="01",
        stage="diagnostics",
        description="Diagnóstico inicial de datasets",
        command=python_script("01_data_diagnostics.py"),
    ),
    PipelineStep(
        step_id="02",
        stage="preprocessing",
        description="Preprocesamiento de partidos",
        command=python_script("02_preprocess_matches.py"),
    ),
    PipelineStep(
        step_id="03",
        stage="preprocessing",
        description="Preprocesamiento de jugadores",
        command=python_script("03_preprocess_players.py"),
    ),
    PipelineStep(
        step_id="04",
        stage="baseline",
        description="Construcción del dataset descriptivo inicial",
        command=python_script("04_build_player_modeling_dataset.py"),
    ),
    PipelineStep(
        step_id="05",
        stage="baseline",
        description="Entrenamiento del baseline descriptivo",
        command=python_script("05_train_player_baseline.py"),
    ),
    PipelineStep(
        step_id="06",
        stage="baseline",
        description="Análisis del baseline descriptivo",
        command=python_script("06_analyze_player_baseline.py"),
    ),
    PipelineStep(
        step_id="07",
        stage="temporal_dataset",
        description="Construcción del dataset temporal",
        command=python_script("07_build_player_temporal_dataset.py"),
    ),
    PipelineStep(
        step_id="08",
        stage="temporal_model",
        description="Entrenamiento y evaluación temporal inicial",
        command=python_script("08_train_player_temporal_model.py"),
    ),
    PipelineStep(
        step_id="09",
        stage="temporal_model",
        description="Generación de predicciones temporales de ejemplo",
        command=python_script("09_generate_player_predictions.py"),
    ),
    PipelineStep(
        step_id="10",
        stage="temporal_model",
        description="Análisis de predicciones temporales de ejemplo",
        command=python_script("10_analyze_player_predictions.py"),
    ),
    PipelineStep(
        step_id="11",
        stage="random_forest_legacy",
        description="Entrenamiento y guardado del modelo Random Forest inicial",
        command=python_script("11_train_and_save_temporal_model.py"),
    ),
    PipelineStep(
        step_id="12",
        stage="random_forest_legacy",
        description="Predicciones actuales iniciales con Random Forest",
        command=python_script("12_predict_laliga_current_players.py"),
    ),
    PipelineStep(
        step_id="13",
        stage="external_evaluation",
        description="Preparación del target externo LaLiga 2025-2026",
        command=python_script("13_prepare_2025_2026_laliga_target.py"),
    ),
    PipelineStep(
        step_id="14",
        stage="external_evaluation",
        description="Evaluación externa inicial 2025-2026",
        command=python_script("14_evaluate_laliga_2025_2026_predictions.py"),
    ),
    PipelineStep(
        step_id="15",
        stage="external_evaluation",
        description="Análisis externo F/M 2025-2026",
        command=python_script("15_analyze_laliga_2025_2026_f_m.py"),
    ),
    PipelineStep(
        step_id="16",
        stage="summaries",
        description="Construcción del resumen inicial de experimentos",
        command=python_script("16_build_experiment_summary.py"),
    ),
    PipelineStep(
        step_id="17",
        stage="summaries",
        description="Generación de figuras iniciales de resultados",
        command=python_script("17_generate_result_figures.py"),
    ),
    PipelineStep(
        step_id="18",
        stage="error_analysis",
        description="Análisis de error por rangos de xG_90",
        command=python_script("18_analyze_error_by_xg_range.py"),
    ),
    PipelineStep(
        step_id="19",
        stage="model_comparison",
        description="Comparación temporal interna de modelos",
        command=python_script("19_compare_temporal_models.py"),
    ),
    PipelineStep(
        step_id="20",
        stage="model_comparison",
        description="Comparación externa 2025-2026 de modelos",
        command=python_script("20_compare_external_2025_2026_models.py"),
    ),
    PipelineStep(
        step_id="21",
        stage="final_model",
        description="Entrenamiento y guardado del modelo final Ridge",
        command=python_script("21_train_and_save_final_ridge_model.py"),
    ),
    PipelineStep(
        step_id="22",
        stage="final_model",
        description="Predicciones actuales con el modelo final Ridge",
        command=python_script("22_predict_laliga_current_players_final_model.py"),
    ),
    PipelineStep(
        step_id="23",
        stage="final_model",
        description="Construcción del resumen final de experimentos",
        command=python_script("23_build_final_experiment_summary.py"),
    ),
]


TEST_STEP = PipelineStep(
    step_id="tests",
    stage="tests",
    description="Ejecución de pruebas unitarias",
    command=[sys.executable, "-m", "pytest"],
)


def get_available_stages() -> list[str]:
    """
    Devuelve las fases disponibles del pipeline en orden de aparición.
    """
    stages = []

    for step in PIPELINE_STEPS:
        if step.stage not in stages:
            stages.append(step.stage)

    stages.append("tests")

    return stages


def validate_required_raw_files() -> None:
    """
    Comprueba que los datasets originales necesarios existen en data/raw.
    """
    missing_files = [
        path for path in REQUIRED_RAW_FILES
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"- {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Faltan datasets originales necesarios en data/raw:\n"
            f"{missing_text}"
        )


def filter_steps_by_stage(
    selected_stage: str,
    include_tests: bool,
) -> list[PipelineStep]:
    """
    Selecciona los pasos que se deben ejecutar según la fase solicitada.
    """
    if selected_stage == "all":
        steps = list(PIPELINE_STEPS)
    elif selected_stage == "tests":
        steps = []
    else:
        steps = [
            step for step in PIPELINE_STEPS
            if step.stage == selected_stage
        ]

    if include_tests:
        if selected_stage in ["all", "tests"]:
            steps.append(TEST_STEP)

    return steps


def run_command(step: PipelineStep) -> tuple[int, float, list[str]]:
    """
    Ejecuta un comando, muestra la salida por consola y devuelve:
    - código de salida,
    - duración,
    - salida completa.
    """
    print("")
    print("=" * 100)
    print(f"Paso {step.step_id}: {step.description}")
    print(f"Fase: {step.stage}")
    print("=" * 100)
    print("Comando:")
    print(" ".join(step.command))
    print("")

    start_time = time.perf_counter()

    process = subprocess.Popen(
        step.command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = []

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

    return_code = process.wait()

    elapsed_seconds = time.perf_counter() - start_time

    print("")
    print(f"Duración: {elapsed_seconds:.2f} segundos")

    if return_code == 0:
        print(f"Resultado: OK")
    else:
        print(f"Resultado: ERROR | Código de salida: {return_code}")

    return return_code, elapsed_seconds, output_lines


def build_log_header(
    selected_stage: str,
    include_tests: bool,
) -> list[str]:
    """
    Construye la cabecera del log de ejecución.
    """
    lines = []

    lines.append("# Log de ejecución del pipeline")
    lines.append("")
    lines.append(f"- Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Python: {sys.version}")
    lines.append(f"- Intérprete: `{sys.executable}`")
    lines.append(f"- Proyecto: `{PROJECT_ROOT}`")
    lines.append(f"- Fase seleccionada: `{selected_stage}`")
    lines.append(f"- Tests incluidos: `{include_tests}`")
    lines.append("")

    return lines


def append_step_log(
    log_lines: list[str],
    step: PipelineStep,
    return_code: int,
    elapsed_seconds: float,
    output_lines: list[str],
) -> None:
    """
    Añade al log la información de un paso ejecutado.
    """
    status = "OK" if return_code == 0 else "ERROR"

    log_lines.append(f"## Paso {step.step_id}: {step.description}")
    log_lines.append("")
    log_lines.append(f"- Fase: `{step.stage}`")
    log_lines.append(f"- Estado: `{status}`")
    log_lines.append(f"- Código de salida: `{return_code}`")
    log_lines.append(f"- Duración: `{elapsed_seconds:.2f}` segundos")
    log_lines.append("")
    log_lines.append("### Comando")
    log_lines.append("")
    log_lines.append("```text")
    log_lines.append(" ".join(step.command))
    log_lines.append("```")
    log_lines.append("")
    log_lines.append("### Salida")
    log_lines.append("")
    log_lines.append("```text")
    log_lines.extend(line.rstrip("\n") for line in output_lines)
    log_lines.append("```")
    log_lines.append("")


def write_log(log_lines: list[str]) -> None:
    """
    Escribe el log en reports/00_pipeline_execution_log.md.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    LOG_OUTPUT_PATH.write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )


def print_available_stages() -> None:
    """
    Muestra por consola las fases disponibles.
    """
    print("Fases disponibles:")
    print("")

    print("- all")

    for stage in get_available_stages():
        print(f"- {stage}")


def parse_arguments() -> argparse.Namespace:
    """
    Define y procesa argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline completo o una fase concreta del sistema "
            "de predicción de rendimiento ofensivo."
        )
    )

    parser.add_argument(
        "--stage",
        default="all",
        choices=["all"] + get_available_stages(),
        help="Fase del pipeline que se desea ejecutar.",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="No ejecuta pytest al final del pipeline completo.",
    )

    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help="Omite la comprobación inicial de datasets en data/raw.",
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Muestra las fases disponibles y termina.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.list_stages:
        print_available_stages()
        return

    include_tests = not args.skip_tests

    if args.stage not in ["all", "tests"]:
        include_tests = False

    if not args.skip_input_check and args.stage == "all":
        validate_required_raw_files()

    steps_to_run = filter_steps_by_stage(
        selected_stage=args.stage,
        include_tests=include_tests,
    )

    if not steps_to_run:
        print("No hay pasos para ejecutar con la configuración indicada.")
        return

    log_lines = build_log_header(
        selected_stage=args.stage,
        include_tests=include_tests,
    )

    total_start_time = time.perf_counter()

    for step in steps_to_run:
        return_code, elapsed_seconds, output_lines = run_command(step)

        append_step_log(
            log_lines=log_lines,
            step=step,
            return_code=return_code,
            elapsed_seconds=elapsed_seconds,
            output_lines=output_lines,
        )

        write_log(log_lines)

        if return_code != 0:
            print("")
            print("Ejecución detenida porque un paso ha fallado.")
            print(f"Revisa el log en: {LOG_OUTPUT_PATH}")
            sys.exit(return_code)

    total_elapsed_seconds = time.perf_counter() - total_start_time

    log_lines.append("## Resumen final")
    log_lines.append("")
    log_lines.append("- Estado global: `OK`")
    log_lines.append(f"- Duración total: `{total_elapsed_seconds:.2f}` segundos")
    log_lines.append("")

    write_log(log_lines)

    print("")
    print("=" * 100)
    print("Pipeline ejecutado correctamente.")
    print(f"Duración total: {total_elapsed_seconds:.2f} segundos")
    print(f"Log generado en: {LOG_OUTPUT_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()