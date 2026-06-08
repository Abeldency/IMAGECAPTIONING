import pandas as pd
import string

CAPTIONS_FILE = r"D:\AI_PROJECTS\ImageCaptioning\dataset\captions.txt"


def load_captions():
    df = pd.read_csv(CAPTIONS_FILE)

    captions_mapping = {}

    for _, row in df.iterrows():

        image = row["image"]
        caption = row["caption"]

        caption = caption.lower()

        caption = caption.translate(
            str.maketrans("", "", string.punctuation)
        )

        caption = " ".join(caption.split())

        caption = "startseq " + caption + " endseq"

        if image not in captions_mapping:
            captions_mapping[image] = []

        captions_mapping[image].append(caption)

    return captions_mapping


if __name__ == "__main__":

    captions = load_captions()

    print("Total Images:", len(captions))

    sample_image = list(captions.keys())[0]

    print("\nSample Image:")
    print(sample_image)

    print("\nCaptions:")

    for cap in captions[sample_image]:
        print(cap)