#!/usr/bin/env python3
import os
import shutil
import random
import argparse
def get_breed_dirs(src_dir, ignore_list=None):
    """Return subdirectories that are not in the ignore_list."""
    ignore_list = ignore_list or []
    return [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and d not in ignore_list
    ]

def split_files(file_list, train_ratio, test_ratio, val_ratio):
    """Shuffle file list and split into three lists."""
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)
    return file_list[:train_end], file_list[train_end:test_end], file_list[test_end:]

def copy_files(files, src_breed_dir, dest_breed_dir):
    os.makedirs(dest_breed_dir, exist_ok=True)
    for file in files:
        src_path = os.path.join(src_breed_dir, file)
        dest_path = os.path.join(dest_breed_dir, file)
        shutil.copy2(src_path, dest_path)

def process_class(src_base, dest_base, class_name, train_ratio, test_ratio, val_ratio, 
                  src_subdir=None, ignore_dirs=None):
    """
    Process one class (e.g., cats or dogs). For cats, src_base is the directory containing breed folders.
    For dogs, if images are stored under a subdirectory (e.g., "images"), provide src_subdir.
    """
    if src_subdir:
        src_dir = os.path.join(src_base, src_subdir)
    else:
        src_dir = src_base

    # Exclude any directories that should be ignored (like existing train/test/val splits)
    ignore_dirs = ignore_dirs or ['train', 'test', 'val']
    breeds = get_breed_dirs(src_dir, ignore_list=ignore_dirs)
    print(f"Processing {class_name} with breeds: {breeds}")
    
    for breed in breeds:
        breed_src_dir = os.path.join(src_dir, breed)
        # List only files (assuming these are images)
        files = [f for f in os.listdir(breed_src_dir) if os.path.isfile(os.path.join(breed_src_dir, f))]
        if not files:
            continue

        train_files, test_files, val_files = split_files(files, train_ratio, test_ratio, val_ratio)
        
        for split, file_list in zip(['train', 'test', 'val'], [train_files, test_files, val_files]):
            dest_breed_dir = os.path.join(dest_base, split, class_name, breed)
            copy_files(file_list, breed_src_dir, dest_breed_dir)
            print(f"Copied {len(file_list)} images of {breed} to {dest_breed_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Reorganize cats and dogs dataset into train, test and val splits with equal breed representation."
    )
    parser.add_argument('--cats', type=str, default='cats',
                        help='Path to the cats folder (each breed in its own subfolder)')
    parser.add_argument('--dogs', type=str, default='dogs',
                        help='Path to the dogs folder. For dogs, images are assumed to be under dogs/images')
    parser.add_argument('--dest', type=str, default='dataset',
                        help='Destination folder where train/test/val folders will be created')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of images for training (default 0.7)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of images for testing (default 0.15)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of images for validation (default 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Ensure ratios sum to 1.0 (allowing for some float imprecision)
    if not abs((args.train_ratio + args.test_ratio + args.val_ratio) - 1.0) < 1e-5:
        parser.error("train_ratio, test_ratio, and val_ratio must sum to 1.0")

    random.seed(args.seed)

    # Create base destination directories
    for split in ['train', 'test', 'val']:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(args.dest, split, cls), exist_ok=True)

    # Process cats: source folders are directly under the cats folder
    process_class(args.cats, args.dest, 'cats', args.train_ratio, args.test_ratio, args.val_ratio)

    # Process dogs: assume images are under dogs/images directory.
    process_class(args.dogs, args.dest, 'dogs', args.train_ratio, args.test_ratio, args.val_ratio,
                  src_subdir='images')

    print("Dataset reorganization complete.")

if __name__ == "__main__":
    main()
