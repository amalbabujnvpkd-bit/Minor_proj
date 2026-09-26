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
import re

def clean_text(text):
    from sklearn.model_selection import train_test_split

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
    df["review"] = df["review"].apply(clean_text)

print(df.head())
    return reviews
    positive_reviews = load_reviews(
    os.path.join(DATA_PATH, "train", "pos"),
    "positive"
)

negative_reviews = load_reviews(
    os.path.join(DATA_PATH, "train", "neg"),
    "negative"
)
all_reviews = positive_reviews + negative_reviews

df = pd.DataFrame(all_reviews)

print(df.head())
print(df["sentiment"].value_counts())
