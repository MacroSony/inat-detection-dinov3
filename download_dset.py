from huggingface_hub import hf_hub_download
import tarfile

train_path = hf_hub_download(repo_id="MacroSony/inat-2017-subset", filename="train_annotations.json", local_dir="./data", repo_type="dataset")
val_path = hf_hub_download(repo_id="MacroSony/inat-2017-subset", filename="val_annotations.json", local_dir="./data", repo_type="dataset")
archive_path = hf_hub_download(repo_id="MacroSony/inat-2017-subset", filename="images.tar.gz", local_dir="./data", repo_type="dataset")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path="data/images")