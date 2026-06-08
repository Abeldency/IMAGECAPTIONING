import os
import pickle
import numpy as np
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add

from tensorflow.keras.callbacks import ModelCheckpoint

from utils import load_captions

# ====================================
# PATHS
# ====================================

FEATURES_FILE = r"D:\AI_PROJECTS\ImageCaptioning\features\features.pkl"
TOKENIZER_FILE = r"D:\AI_PROJECTS\ImageCaptioning\models\tokenizer.pkl"
MODEL_FILE = r"D:\AI_PROJECTS\ImageCaptioning\models\caption_model.keras"

# ====================================
# LOAD DATA
# ====================================

print("Loading captions...")
captions = load_captions()

print("Loading features...")
with open(FEATURES_FILE, "rb") as f:
    features = pickle.load(f)

print("Captions:", len(captions))
print("Features:", len(features))

# ====================================
# TOKENIZER
# ====================================

all_captions = []

for caps in captions.values():
    all_captions.extend(caps)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1

max_length = max(
    len(caption.split())
    for caption in all_captions
)

print("Vocabulary Size:", vocab_size)
print("Max Length:", max_length)

with open(TOKENIZER_FILE, "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved.")

# ====================================
# GENERATOR
# ====================================

class DataGenerator(Sequence):

    def __init__(
        self,
        captions,
        features,
        tokenizer,
        max_length,
        vocab_size,
        batch_size=32
    ):

        self.image_ids = list(captions.keys())

        self.captions = captions
        self.features = features

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.vocab_size = vocab_size

        self.batch_size = batch_size

    def __len__(self):

        return int(
            np.ceil(
                len(self.image_ids) / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch_ids = self.image_ids[
            index*self.batch_size:
            (index+1)*self.batch_size
        ]

        X1 = []
        X2 = []
        y = []

        for image_id in batch_ids:

            feature = self.features[image_id][0]

            for caption in self.captions[image_id]:

                seq = self.tokenizer.texts_to_sequences(
                    [caption]
                )[0]

                for i in range(1, len(seq)):

                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences(
                        [in_seq],
                        maxlen=self.max_length
                    )[0]

                    out_seq = to_categorical(
                        out_seq,
                        num_classes=self.vocab_size
                    )

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return (
            (
                np.array(X1),
                np.array(X2)
            ),
            np.array(y)
        )

# ====================================
# MODEL
# ====================================

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(
    vocab_size,
    256,
    mask_zero=True
)(inputs2)

se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])

decoder2 = Dense(
    256,
    activation='relu'
)(decoder1)

outputs = Dense(
    vocab_size,
    activation='softmax'
)(decoder2)

model = Model(
    inputs=[inputs1, inputs2],
    outputs=outputs
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)

model.summary()

# ====================================
# TRAINING
# ====================================

generator = DataGenerator(
    captions,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size=32
)

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="loss",
    save_best_only=True,
    verbose=1
)

print("\nStarting Training...\n")

model.fit(
    generator,
    epochs=10,
    callbacks=[checkpoint]
)

print("\nTraining Complete.")