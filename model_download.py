from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='Qwen/Qwen3.5-0.8B-Base',
    local_dir='./model/qwen3.6-0.8b'
)