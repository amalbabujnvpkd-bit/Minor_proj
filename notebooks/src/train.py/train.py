import os
import pandas as pd
DATA_PATH = "../data/aclImdb"
def load_reviews(folder_path, label):
    reviews = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        reviews.append({
            "review": text,
            "sentiment": label
        })

    return reviews
