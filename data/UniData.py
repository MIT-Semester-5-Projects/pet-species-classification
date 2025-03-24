import os
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


class UniversalDataset(Dataset):
    def __init__(
        self,
        subset: Literal["test", "train", "val"] = "train",
        option: Literal["All", "Cat", "Dog"] = "All",
        transform=None,
    ) -> None:
        self.rootdir = os.path.join("../data", subset)
        self.dog_dir = os.path.join(self.rootdir, "dogs")
        self.cat_dir = os.path.join(self.rootdir, "cats")
        self.transform = transform
        self.option = option

        self.dog_img_paths = []
        self.cat_img_paths = []
        
        # For species classification, these labels remain.
        self.species_labels = {0: "dog", 1: "cat"}

        # Getting all dog image paths
        for root, dirs, files in os.walk(self.dog_dir):
            for file in files:
                self.dog_img_paths.append(os.path.join(root, file))

        # Getting all cat image paths
        for root, dirs, files in os.walk(self.cat_dir):
            for file in files:
                self.cat_img_paths.append(os.path.join(root, file))

        # Build breed mappings if option is "Cat" or "Dog"
        if self.option == "Cat":
            self.cat_breeds = set()
            for path in self.cat_img_paths:
                # Assumes folder structure: ../data/<subset>/cats/<breed>/image.jpg
                breed = os.path.basename(os.path.dirname(path))
                self.cat_breeds.add(breed)
            self.cat_breeds = sorted(list(self.cat_breeds))
            self.cat_breed2idx = {b: i for i, b in enumerate(self.cat_breeds)}
            # Optionally, you can print or log the mapping:
            # print("Cat breed mapping:", self.cat_breed2idx)
        elif self.option == "Dog":
            self.dog_breeds = set()
            for path in self.dog_img_paths:
                # Assumes folder structure: ../data/<subset>/dogs/<breed>/image.jpg
                breed = os.path.basename(os.path.dirname(path))
                self.dog_breeds.add(breed)
            self.dog_breeds = sorted(list(self.dog_breeds))
            self.dog_breed2idx = {b: i for i, b in enumerate(self.dog_breeds)}
            # print("Dog breed mapping:", self.dog_breed2idx)

    def __len__(self):
        if self.option == "All":
            return len(self.dog_img_paths) + len(self.cat_img_paths)
        elif self.option == "Cat":
            return len(self.cat_img_paths)
        elif self.option == "Dog":
            return len(self.dog_img_paths)
        else:
            raise ValueError(f"Invalid option {self.option}")

    def __getitem__(self, index):
        # Option "All": combine dog and cat images and use species-level labels.
        if self.option == "All":
            num_dogs = len(self.dog_img_paths)
            if index < num_dogs:
                img_path = self.dog_img_paths[index]
                label = 0  # dog
            else:
                img_path = self.cat_img_paths[index - num_dogs]
                label = 1  # cat

        # Option "Cat": only cat images and breed as label
        elif self.option == "Cat":
            img_path = self.cat_img_paths[index]
            # Extract breed from the directory name (assumes ../data/.../cats/<breed>/image.jpg)
            breed = os.path.basename(os.path.dirname(img_path))
            label = self.cat_breed2idx[breed]

        # Option "Dog": only dog images and breed as label
        elif self.option == "Dog":
            img_path = self.dog_img_paths[index]
            # Extract breed from the directory name (assumes ../data/.../dogs/<breed>/image.jpg)
            breed = os.path.basename(os.path.dirname(img_path))
            label = self.dog_breed2idx[breed]

        else:
            raise ValueError(
                f"Invalid option {self.option}. Use 'All', 'Cat', or 'Dog'."
            )

        # Open the image and convert to RGB
        img = Image.open(img_path).convert("RGB")

        # Apply transformation if provided
        if self.transform:
            img = self.transform(img)

        return img, label
