from huggingface_hub import HfApi

api = HfApi()
repo_id = "MacroSony/inat-2017-subset"

api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

files_to_upload = [
    ("data/README.md", "README.md"),
    ("data/subset_train_bboxes.json", "train_annotations.json"),
    ("data/subset_val_bboxes.json", "val_annotations.json"),
    ("data/images.tar.gz", "images.tar.gz"),
]

for local_path, path_in_repo in files_to_upload:
    print(f"Uploading {local_path}...")
    api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
)