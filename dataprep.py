import os
import json
import random
import requests
import tarfile
from collections import defaultdict

INAT_SUPERCATEGORIES = [
    "Actinopterygii",
    "Amphibia",
    "Animalia",
    "Arachnida",
    "Aves",
    "Insecta",
    "Mammalia",
    "Mollusca",
    "Reptilia",
]

def download_and_unpack_annotations():
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists("data/val_bboxes.zip"):
        print("Downloading val_bboxes.zip...")
        response = requests.get("https://ml-inat-competition-datasets.s3.amazonaws.com/2017/val_2017_bboxes.zip")
        with open("data/val_bboxes.zip", "wb") as f:
            f.write(response.content)

    if not os.path.exists("data/train_bboxes.zip"):
        print("Downloading train_bboxes.zip...")
        response = requests.get("https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip")
        with open("data/train_bboxes.zip", "wb") as f:
            f.write(response.content)
    
    import shutil
    if not os.path.exists("data/val_bboxes/val_2017_bboxes.json"):
        shutil.unpack_archive("data/val_bboxes.zip", "data/val_bboxes")
    if not os.path.exists("data/train_bboxes/train_2017_bboxes.json"):
        shutil.unpack_archive("data/train_bboxes.zip", "data/train_bboxes")

def create_subset(json_path, output_json_path, samples_per_supercat=1000):
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map category_id to supercategory
    cat_to_supercat = {cat['id']: cat['supercategory'] for cat in data['categories']}
    
    # Group images by supercategory
    # Note: an image might have multiple annotations, but we group by the primary annotation's supercat
    supercat_to_images = defaultdict(set)
    for ann in data['annotations']:
        cat_id = ann['category_id']
        supercat = cat_to_supercat[cat_id]
        supercat_to_images[supercat].add(ann['image_id'])
        
    sampled_image_ids = set()
    print("Sampling images per supercategory:")
    for supercat, img_ids in supercat_to_images.items():
        img_ids_list = list(img_ids)
        if len(img_ids_list) > samples_per_supercat:
            sampled = random.sample(img_ids_list, samples_per_supercat)
        else:
            sampled = img_ids_list
        sampled_image_ids.update(sampled)
        print(f"  - {supercat}: {len(sampled)} images")
        
    # Map image_id to its dict
    img_dict = {img['id']: img for img in data['images']}
        
    # Filter images
    new_images = [img_dict[img_id] for img_id in sampled_image_ids]
    
    # Filter annotations
    new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in sampled_image_ids]
    
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f)
        
    # Build a set of exactly what the file_name is in the JSON
    filenames_to_extract = set(img['file_name'] for img in new_images)
    print(f"\nTotal sampled images: {len(filenames_to_extract)}")
    print(f"Saved subset annotations to {output_json_path}")
    return filenames_to_extract

def remap_categories_to_supercategories(json_path, output_json_path=None):
    """Collapse iNaturalist species categories into contiguous supercategory IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    supercat_to_id = {name: idx for idx, name in enumerate(INAT_SUPERCATEGORIES)}
    cat_to_supercat = {cat["id"]: cat["supercategory"] for cat in data["categories"]}

    for ann in data["annotations"]:
        supercat = cat_to_supercat[ann["category_id"]]
        ann["category_id"] = supercat_to_id[supercat]

    data["categories"] = [
        {"id": idx, "name": name, "supercategory": "organism"}
        for idx, name in enumerate(INAT_SUPERCATEGORIES)
    ]

    output_json_path = output_json_path or json_path
    with open(output_json_path, "w") as f:
        json.dump(data, f)

    print(f"Remapped categories to {len(INAT_SUPERCATEGORIES)} supercategories in {output_json_path}")

def stream_and_extract_tar(url, filenames_to_extract, extract_path="data/images"):
    # Create the output directory
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"\nConnecting to {url} to stream the tar archive...")
    # stream=True prevents loading the 165GB file into memory/disk
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    extracted_count = 0
    total_to_extract = len(filenames_to_extract)
    
    # tarfile can read directly from the streaming raw socket (fileobj)
    # mode 'r|gz' explicitly tells tarfile this is a streamed gzip archive
    with tarfile.open(fileobj=response.raw, mode='r|gz') as tar:
        for member in tar:
            # iNat member names might start with './' or just match the JSON exactly
            member_name = member.name.lstrip("./")
            
            if member_name in filenames_to_extract:
                # Extract this specific member directly to our extract_path
                tar.extract(member, path=extract_path)
                filenames_to_extract.remove(member_name) # Remove to speed up lookups
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    print(f"Extracted {extracted_count}/{total_to_extract} images...")
                    
            # If we've found all the files we needed, stop the stream early
            if not filenames_to_extract:
                print("All requested images extracted. Stopping stream early to save bandwidth.")
                break
                
    print(f"Extraction complete. Total extracted: {extracted_count}")

if __name__ == "__main__":
    # Ensure annotations exist
    download_and_unpack_annotations()
    
    # Process Train subset
    train_json = 'data/train_bboxes/train_2017_bboxes.json'
    train_out = 'data/subset_train_bboxes.json'
    url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz"
    
    filenames = create_subset(train_json, train_out, samples_per_supercat=1000)
    remap_categories_to_supercategories(train_out)
    
    print("\nStarting streaming extraction. This may take a while depending on your internet speed and where the files are in the archive...")
    stream_and_extract_tar(url, filenames)
