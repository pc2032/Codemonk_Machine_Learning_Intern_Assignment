import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load data
styles = pd.read_csv('fashion-dataset/styles.csv', on_bad_lines='skip')
data = styles[['id', 'masterCategory', 'gender', 'season', 'baseColour']].dropna()
image_dir = 'fashion-dataset/images/'

def load_data(df, img_size=(224, 224), max_samples=500):
    X = []
    y_category = []
    y_gender = []
    y_season = []
    y_color = []
    for _, row in df.sample(max_samples).iterrows():
        img_path = os.path.join(image_dir, f"{row['id']}.jpg")
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img) / 255.0
            X.append(img)
            y_category.append(row['masterCategory'])
            y_gender.append(row['gender'])
            y_season.append(row['season'])
            y_color.append(row['baseColour'])
    return np.array(X), np.array(y_category), np.array(y_gender), np.array(y_season), np.array(y_color)

X, y_cat, y_gen, y_sea, y_col = load_data(data)

# Turn words into numbers
le_cat = LabelEncoder()
le_gen = LabelEncoder()
le_sea = LabelEncoder()
le_col = LabelEncoder()

y_cat = le_cat.fit_transform(y_cat)
y_gen = le_gen.fit_transform(y_gen)
y_sea = le_sea.fit_transform(y_sea)
y_col = le_col.fit_transform(y_col)

# Split data
X_train, X_test, y_cat_train, y_cat_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
_, _, y_gen_train, y_gen_test = train_test_split(X, y_gen, test_size=0.2, random_state=42)
_, _, y_sea_train, y_sea_test = train_test_split(X, y_sea, test_size=0.2, random_state=42)
_, _, y_col_train, y_col_test = train_test_split(X, y_col, test_size=0.2, random_state=42)

# Build the model
input_layer = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

out_category = Dense(len(le_cat.classes_), activation='softmax', name='category')(x)
out_gender = Dense(len(le_gen.classes_), activation='softmax', name='gender')(x)
out_season = Dense(len(le_sea.classes_), activation='softmax', name='season')(x)
out_color = Dense(len(le_col.classes_), activation='softmax', name='color')(x)

model = Model(inputs=input_layer, outputs=[out_category, out_gender, out_season, out_color])
model.compile(optimizer='adam',
              loss={'category': 'sparse_categorical_crossentropy',
                    'gender': 'sparse_categorical_crossentropy',
                    'season': 'sparse_categorical_crossentropy',
                    'color': 'sparse_categorical_crossentropy'},
              metrics={'category': 'accuracy',
                       'gender': 'accuracy',
                       'season': 'accuracy',
                       'color': 'accuracy'})

# Train it
model.fit(X_train, 
          {'category': y_cat_train, 'gender': y_gen_train, 'season': y_sea_train, 'color': y_col_train},
          epochs=5, batch_size=32, validation_data=(X_test, {'category': y_cat_test, 'gender': y_gen_test, 'season': y_sea_test, 'color': y_col_test}))

# Save it
model.save('fashion_model.h5')
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'category': le_cat, 'gender': le_gen, 'season': le_sea, 'color': le_col}, f)

print("Model saved!")