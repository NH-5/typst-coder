from huggingface_hub import snapshot_download
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent

snapshot_download(
    repo_id='Qwen/Qwen3.5-0.8B-Base',
    local_dir=PROJECT_PATH / 'model/qwen3.5-0.8b-Base'
)
