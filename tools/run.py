from datetime import datetime as dt
from pathlib import Path
import yaml
import click

from pipelines import pdf_ingestion_pipeline


@click.command()
@click.option(
    "--run-extract-pdf",
    is_flag=True,
    default=False,
    help="Run PDF extraction pipeline",
)
@click.option(
    "--pdf-config-filename",
    default="pdf_ingestion.yaml",
    help="PDF ingestion config filename",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable ZenML cache",
)
def main(
    run_extract_pdf: bool,
    pdf_config_filename: str,
    no_cache: bool,
):
    assert run_extract_pdf, "Please specify --run-extract-pdf"

    pipeline_args = {
        "enable_cache": not no_cache,
    }

    root_dir = Path(__file__).resolve().parent.parent

    if run_extract_pdf:
        pipeline_args["config_path"] = root_dir / "configs" / pdf_config_filename
        assert pipeline_args[
            "config_path"
        ].exists(), f"Config file not found: {pipeline_args['config_path']}"

        pipeline_args["run_name"] = (
            f"pdf_ingestion_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )

        config = yaml.safe_load(pipeline_args["config_path"].read_text())
        pdf_path = config["pdf_path"]

        run_args_pdf = {
            "pdf_path": pdf_path,
        }

        pdf_ingestion_pipeline.with_options(**pipeline_args)(**run_args_pdf)


if __name__ == "__main__":
    main()
