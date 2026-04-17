from huggingface_hub import hf_hub_download


hf_hub_download(
    repo_id='TechxGenus/Typst-Train',
    filename='typst_train.json',
    repo_type='dataset',
    local_dir='./data/raw/train/'
)
hf_hub_download(
    repo_id='TechxGenus/Typst-Test',
    filename='typst_test.json',
    repo_type='dataset',
    local_dir='./data/raw/test/'
)