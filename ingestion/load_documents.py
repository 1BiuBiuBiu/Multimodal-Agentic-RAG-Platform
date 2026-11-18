"""Legacy entry point: share the MinerU ingestion and hybrid index pipeline."""

from ingestion.ingest import run_ingestion


if __name__ == "__main__":
    run_ingestion()
