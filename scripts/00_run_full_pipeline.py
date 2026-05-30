import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TESTS_DIR = PROJECT_ROOT / "tests"

LOG_OUTPUT_PATH = REPORTS_DIR / "00_pipeline_execution_log.md"


REQUIRED_INPUT_FILES = [
    RAW_DATA_DIR / "matches_dataset.csv",
    RAW_DATA_DIR / "all_players_normalized.csv",
    RAW_DATA_DIR / "players_laliga.csv",
    RAW_DATA_DIR / "laliga_players_2526.csv",
]


@dataclass(frozen=True)
class PipelineStep:
    """
    Representa un paso ejecutable del pipeline.
    """
    stage: str
    script: str
    description: str


PIPELINE_STEPS = [
    PipelineStep(
        stage="diagnostics",
        script="01_data_diagnostics.py",
        description="Diagnóstico inicial de los datasets raw.",
    ),
    PipelineStep(
        stage="preprocessing",
        script="02_preprocess_matches.py",
        description="Limpieza del dataset histórico de partidos.",
    ),
    PipelineStep(
        stage="preprocessing",
        script="03_preprocess_players.py",
        description="Limpieza de datasets históricos y actuales de jugadores.",
    ),
    PipelineStep(
        stage="preprocessing",
        script="13_prepare_2025_2026_laliga_target.py",
        description="Preparación del target real 2025-2026 de LaLiga.",
    ),
    PipelineStep(
        stage="baseline",
        script="04_build_player_modeling_dataset.py",
        description="Construcción del dataset descriptivo inicial de modelado.",
    ),
    PipelineStep(
        stage="baseline",
        script="05_train_player_baseline.py",
        description="Entrenamiento del baseline descriptivo inicial.",
    ),
    PipelineStep(
        stage="baseline",
        script="06_analyze_player_baseline.py",
        description="Análisis de importancia de variables del baseline.",
    ),
    PipelineStep(
        stage="temporal_dataset",
        script="07_build_player_temporal_dataset.py",
        description="Construcción del dataset temporal jugador-temporada.",
    ),
    PipelineStep(
        stage="temporal_model",
        script="08_train_player_temporal_model.py",
        description="Entrenamiento y evaluación temporal inicial.",
    ),
    PipelineStep(
        stage="temporal_model",
        script="09_generate_player_predictions.py",
        description="Generación de predicciones temporales de ejemplo.",
    ),
    PipelineStep(
        stage="temporal_model",
        script="10_analyze_player_predictions.py",
        description="Análisis cualitativo de predicciones individuales.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="11_train_and_save_temporal_model.py",
        description="Entrenamiento y guardado del modelo Random Forest legacy.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="12_predict_laliga_current_players.py",
        description="Predicciones prospectivas legacy con Random Forest.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="14_evaluate_laliga_2025_2026_predictions.py",
        description="Evaluación externa legacy 2025-2026 con Random Forest.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="15_analyze_laliga_2025_2026_f_m.py",
        description="Evaluación externa legacy restringida a F/M.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="16_build_experiment_summary.py",
        description="Resumen experimental legacy previo a la selección final.",
    ),
    PipelineStep(
        stage="random_forest_legacy",
        script="17_generate_result_figures.py",
        description="Generación de figuras legacy de resultados.",
    ),
    PipelineStep(
        stage="error_analysis",
        script="18_analyze_error_by_xg_range.py",
        description="Análisis del error por rangos de xG_90 real.",
    ),
    PipelineStep(
        stage="model_comparison",
        script="19_compare_temporal_models.py",
        description="Comparación temporal interna de modelos candidatos.",
    ),
    PipelineStep(
        stage="model_comparison",
        script="20_compare_external_2025_2026_models.py",
        description=(
            "Comparación externa de modelos para xG_90 y evaluación "
            "multi-métrica con Ridge."
        ),
    ),
    PipelineStep(
        stage="final_model",
        script="21_train_and_save_final_ridge_model.py",
        description="Entrenamiento y guardado de modelos Ridge finales multi-métrica.",
    ),
    PipelineStep(
        stage="final_model",
        script="22_predict_laliga_current_players_final_model.py",
        description="Predicciones prospectivas multi-métrica con Ridge.",
    ),
    PipelineStep(
        stage="final_model",
        script="23_build_final_experiment_summary.py",
        description="Resumen final de experimentos multi-métrica.",
    ),
]


STAGE_ORDER = [
    "diagnostics",
    "preprocessing",
    "baseline",
    "temporal_dataset",
    "temporal_model",
    "random_forest_legacy",
    "error_analysis",
    "model_comparison",
    "final_model",
    "tests",
]


STAGE_DESCRIPTIONS = {
    "all": "Ejecuta el pipeline completo.",
    "diagnostics": "Diagnóstico inicial de datos raw.",
    "preprocessing": "Limpieza de datos y preparación del target 2025-2026.",
    "baseline": "Baseline descriptivo inicial de jugadores.",
    "temporal_dataset": "Construcción del dataset temporal jugador-temporada.",
    "temporal_model": "Evaluación temporal inicial y análisis de predicciones.",
    "random_forest_legacy": "Experimentos legacy con Random Forest.",
    "error_analysis": "Análisis del error por rangos de xG_90.",
    "model_comparison": "Comparación de modelos y evaluación externa multi-métrica.",
    "final_model": "Entrenamiento, predicción y resumen final multi-métrica con Ridge.",
    "tests": "Ejecución de tests automatizados con pytest.",
}


EXPECTED_OUTPUTS_BY_STAGE = {
    "diagnostics": [
        REPORTS_DIR / "01_data_diagnostics.md",
    ],
    "preprocessing": [
        PROCESSED_DATA_DIR / "matches_clean.csv",
        PROCESSED_DATA_DIR / "historical_players_clean.csv",
        PROCESSED_DATA_DIR / "laliga_players_all_clean.csv",
        PROCESSED_DATA_DIR / "laliga_players_latest_clean.csv",
        PROCESSED_DATA_DIR / "laliga_2025_2026_target_clean.csv",
    ],
    "baseline": [
        PROCESSED_DATA_DIR / "player_modeling_dataset.csv",
        PROCESSED_DATA_DIR / "player_baseline_evaluation.csv",
        PROCESSED_DATA_DIR / "player_feature_importance.csv",
        REPORTS_DIR / "06_player_baseline_analysis.md",
    ],
    "temporal_dataset": [
        PROCESSED_DATA_DIR / "player_temporal_modeling_dataset.csv",
    ],
    "temporal_model": [
        PROCESSED_DATA_DIR / "player_temporal_evaluation.csv",
        PROCESSED_DATA_DIR / "player_temporal_feature_importance.csv",
        PROCESSED_DATA_DIR / "player_temporal_predictions_without_previous_xg.csv",
        PROCESSED_DATA_DIR / "player_temporal_predictions_f_m_sample_15.csv",
        REPORTS_DIR / "08_player_temporal_model_report.md",
        REPORTS_DIR / "10_player_prediction_examples.md",
    ],
    "random_forest_legacy": [
        MODELS_DIR / "random_forest_temporal_without_previous_xg.joblib",
        MODELS_DIR / "random_forest_temporal_without_previous_xg_metadata.json",
        PROCESSED_DATA_DIR / "laliga_current_player_predictions.csv",
        PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions.csv",
        PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation.csv",
        PROCESSED_DATA_DIR / "laliga_2025_2026_prediction_evaluation_f_m.csv",
        PROCESSED_DATA_DIR / "experiment_summary.csv",
        REPORTS_DIR / "14_laliga_2025_2026_prediction_evaluation.md",
        REPORTS_DIR / "15_laliga_2025_2026_prediction_evaluation_f_m.md",
        REPORTS_DIR / "16_experiment_summary.md",
        REPORTS_DIR / "17_result_figures_index.md",
    ],
    "error_analysis": [
        PROCESSED_DATA_DIR / "error_by_xg_range.csv",
        REPORTS_DIR / "18_error_by_xg_range.md",
        FIGURES_DIR / "error_by_xg_range.png",
    ],
    "model_comparison": [
        PROCESSED_DATA_DIR / "temporal_model_comparison.csv",
        PROCESSED_DATA_DIR / "temporal_model_comparison_by_xg_range.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_model_comparison.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_f_m.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_model_predictions.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_model_comparison_by_xg_range.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_predictions.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_predictions_long.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison.csv",
        PROCESSED_DATA_DIR / "external_2025_2026_multi_metric_comparison_f_m.csv",
        REPORTS_DIR / "19_temporal_model_comparison.md",
        REPORTS_DIR / "20_external_2025_2026_model_comparison.md",
        FIGURES_DIR / "external_2025_2026_multi_metric_mae.png",
        FIGURES_DIR / "external_2025_2026_multi_metric_r2.png",
    ],
    "final_model": [
        MODELS_DIR / "ridge_temporal_xG_90.joblib",
        MODELS_DIR / "ridge_temporal_goals_90.joblib",
        MODELS_DIR / "ridge_temporal_assists_90.joblib",
        MODELS_DIR / "ridge_temporal_xA_90.joblib",
        MODELS_DIR / "ridge_temporal_multi_metric_metadata.json",
        MODELS_DIR / "ridge_temporal_without_previous_xg.joblib",
        MODELS_DIR / "ridge_temporal_without_previous_xg_metadata.json",
        PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge.csv",
        PROCESSED_DATA_DIR / "laliga_current_player_predictions_final_ridge_long.csv",
        PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_final_ridge.csv",
        PROCESSED_DATA_DIR / "laliga_current_top_25_f_m_predictions_by_metric_final_ridge.csv",
        PROCESSED_DATA_DIR / "final_experiment_summary.csv",
        PROCESSED_DATA_DIR / "final_multi_metric_summary.csv",
        PROCESSED_DATA_DIR / "final_prospective_prediction_summary.csv",
        REPORTS_DIR / "22_laliga_current_predictions_final_model.md",
        REPORTS_DIR / "23_final_experiment_summary.md",
        FIGURES_DIR / "laliga_current_top_25_final_ridge_xG_90.png",
        FIGURES_DIR / "laliga_current_top_25_final_ridge_goals_90.png",
        FIGURES_DIR / "laliga_current_top_25_final_ridge_assists_90.png",
        FIGURES_DIR / "laliga_current_top_25_final_ridge_xA_90.png",
    ],
}


# ---------------------------------------------------------------------------
# Utilidades de consola y logging
# ---------------------------------------------------------------------------


def format_relative_path(path: Path) -> str:
    """
    Devuelve una ruta relativa al proyecto cuando es posible.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def initialize_log() -> None:
    """
    Inicializa el informe de ejecución del pipeline.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Log de ejecución del pipeline",
        "",
        f"- Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Directorio del proyecto: `{PROJECT_ROOT}`",
        "",
    ]

    LOG_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def append_log(lines: list[str]) -> None:
    """
    Añade líneas al informe de ejecución.
    """
    with LOG_OUTPUT_PATH.open("a", encoding="utf-8") as file:
        file.write("\n" + "\n".join(lines) + "\n")


def print_header(message: str) -> None:
    """
    Imprime un encabezado visible en consola.
    """
    separator = "=" * 80
    print("\n" + separator)
    print(message)
    print(separator)


def print_stage_list() -> None:
    """
    Muestra las fases disponibles.
    """
    print("Fases disponibles:\n")

    for stage_name in ["all"] + STAGE_ORDER:
        print(f"- {stage_name}: {STAGE_DESCRIPTIONS[stage_name]}")


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------


def check_required_inputs() -> None:
    """
    Comprueba que existen los archivos raw mínimos para ejecutar el pipeline.
    """
    missing_files = [
        path for path in REQUIRED_INPUT_FILES
        if not path.exists()
    ]

    if missing_files:
        formatted_missing_files = "\n".join(
            f"- {format_relative_path(path)}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Faltan archivos de entrada requeridos:\n"
            f"{formatted_missing_files}"
        )

    print("Archivos de entrada requeridos encontrados.")


def check_stage_outputs(stage_name: str) -> list[Path]:
    """
    Comprueba salidas esperadas de una fase concreta.
    """
    expected_outputs = EXPECTED_OUTPUTS_BY_STAGE.get(stage_name, [])

    missing_outputs = [
        path for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        print("Aviso: faltan algunas salidas esperadas de la fase:")
        for path in missing_outputs:
            print(f"- {format_relative_path(path)}")
    else:
        if expected_outputs:
            print("Salidas esperadas de la fase encontradas.")

    return missing_outputs


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------


def get_steps_for_stage(stage_name: str) -> list[PipelineStep]:
    """
    Devuelve los pasos correspondientes a una fase.
    """
    if stage_name == "all":
        return PIPELINE_STEPS

    return [
        step for step in PIPELINE_STEPS
        if step.stage == stage_name
    ]


def run_command(command: list[str], description: str) -> subprocess.CompletedProcess:
    """
    Ejecuta un comando y registra salida estándar y errores.
    """
    command_as_text = " ".join(command)

    print_header(description)
    print(f"Comando: {command_as_text}")

    append_log(
        [
            f"## {description}",
            "",
            f"```powershell\n{command_as_text}\n```",
            "",
        ]
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout)
        append_log(
            [
                "### Salida estándar",
                "",
                f"```text\n{result.stdout.strip()}\n```",
                "",
            ]
        )

    if result.stderr:
        print(result.stderr)
        append_log(
            [
                "### Errores / advertencias",
                "",
                f"```text\n{result.stderr.strip()}\n```",
                "",
            ]
        )

    if result.returncode != 0:
        append_log(
            [
                f"**Estado:** error con código `{result.returncode}`.",
                "",
            ]
        )
        raise RuntimeError(
            f"El comando ha fallado con código {result.returncode}: "
            f"{command_as_text}"
        )

    append_log(["**Estado:** ejecutado correctamente.", ""])

    return result


def run_pipeline_steps(
    steps: list[PipelineStep],
    validate_outputs: bool,
) -> None:
    """
    Ejecuta una lista de pasos del pipeline.
    """
    current_stage = None

    for step in steps:
        if step.stage != current_stage:
            current_stage = step.stage
            print_header(f"Fase: {current_stage}")
            append_log([f"# Fase: {current_stage}", ""])

        script_path = SCRIPTS_DIR / step.script

        if not script_path.exists():
            raise FileNotFoundError(
                f"No se ha encontrado el script: {script_path}"
            )

        command = [
            sys.executable,
            str(script_path),
        ]

        run_command(
            command=command,
            description=step.description,
        )

    if validate_outputs:
        completed_stages = sorted(
            {step.stage for step in steps},
            key=lambda stage: STAGE_ORDER.index(stage),
        )

        for stage_name in completed_stages:
            missing_outputs = check_stage_outputs(stage_name)

            if missing_outputs:
                append_log(
                    [
                        f"## Aviso de salidas faltantes en `{stage_name}`",
                        "",
                        *[
                            f"- `{format_relative_path(path)}`"
                            for path in missing_outputs
                        ],
                        "",
                    ]
                )


def run_tests() -> None:
    """
    Ejecuta la suite de tests con pytest.
    """
    if not TESTS_DIR.exists():
        print("No se ha encontrado el directorio tests/. Se omite la fase de tests.")
        append_log(["# Fase: tests", "", "No existe `tests/`. Fase omitida.", ""])
        return

    print_header("Fase: tests")
    append_log(["# Fase: tests", ""])

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR),
    ]

    run_command(
        command=command,
        description="Ejecución de tests automatizados con pytest.",
    )


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construye el parser de argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline completo o una fase específica del TFG de "
            "predicción de rendimiento ofensivo de jugadores."
        )
    )

    parser.add_argument(
        "--stage",
        choices=["all"] + STAGE_ORDER,
        default="all",
        help="Fase del pipeline que se desea ejecutar.",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Omite la ejecución de tests cuando se ejecuta el pipeline completo.",
    )

    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help="Omite la comprobación inicial de archivos raw requeridos.",
    )

    parser.add_argument(
        "--skip-output-check",
        action="store_true",
        help="Omite la comprobación de salidas esperadas por fase.",
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Lista las fases disponibles y termina la ejecución.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.list_stages:
        print_stage_list()
        return

    initialize_log()

    print_header("Inicio del pipeline")
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Fase solicitada: {args.stage}")
    print(f"Log: {LOG_OUTPUT_PATH}")

    append_log(
        [
            "## Configuración de ejecución",
            "",
            f"- Fase solicitada: `{args.stage}`",
            f"- Omitir tests: `{args.skip_tests}`",
            f"- Omitir comprobación de inputs: `{args.skip_input_check}`",
            f"- Omitir comprobación de outputs: `{args.skip_output_check}`",
            "",
        ]
    )

    if not args.skip_input_check:
        check_required_inputs()

    validate_outputs = not args.skip_output_check

    if args.stage == "tests":
        run_tests()
    elif args.stage == "all":
        run_pipeline_steps(
            steps=PIPELINE_STEPS,
            validate_outputs=validate_outputs,
        )

        if not args.skip_tests:
            run_tests()
    else:
        selected_steps = get_steps_for_stage(args.stage)

        if not selected_steps:
            raise ValueError(
                f"La fase `{args.stage}` no contiene pasos ejecutables."
            )

        run_pipeline_steps(
            steps=selected_steps,
            validate_outputs=validate_outputs,
        )

    append_log(
        [
            "# Ejecución completada",
            "",
            f"Finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
    )

    print_header("Pipeline finalizado correctamente")
    print(f"Log generado en: {LOG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
