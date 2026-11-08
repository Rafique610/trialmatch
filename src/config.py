"""Project settings. Every path, model id and env value is read here, nowhere else."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TM_", env_file=".env", extra="ignore")

    dataset_id: str = "ncbi/TrialGPT-Criterion-Annotations"
    dataset_revision: str = "1cfcafde94a1560a33b4addc1638664fe28fc059"  # pinned so the split never shifts
    target_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    draft_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_repo_id: str = "Aduhere/trialmatch-qwen2.5-1.5b-qlora"  # chosen adapter (TRIA-7)

    split_seed: int = 42
    split_sizes: tuple[int, int, int] = (37, 8, 8)

    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"
    checkpoints_dir: Path = ROOT / "checkpoints"
    models_dir: Path = ROOT / "models"  # optional local copies (git-ignored), used before the HF cache

    def model_source(self, repo_id: str) -> str:
        """models/<name> if a full local copy exists, else the Hugging Face id."""
        local = self.models_dir / repo_id.split("/")[-1]
        return str(local) if (local / "config.json").exists() else repo_id


settings = Settings()
