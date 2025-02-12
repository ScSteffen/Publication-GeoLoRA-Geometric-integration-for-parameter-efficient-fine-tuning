from huggingface_hub import snapshot_download

local_dir = '/dog-example'   ## update to your path
# local_dir = './'
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
