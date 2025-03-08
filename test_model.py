import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import os

# Load the model
model = tf.keras.models.load_model('fashion_model.h5')
with open('label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)

def predict_image(img_path, model, le_dict):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return {
        'Category': le_dict['category'].inverse_transform([np.argmax(preds[0])])[0],
        'Gender': le_dict['gender'].inverse_transform([np.argmax(preds[1])])[0],
        'Season': le_dict['season'].inverse_transform([np.argmax(preds[2])])[0],
        'Color': le_dict['color'].inverse_transform([np.argmax(preds[3])])[0]
    }

# Test the pictures
test_dir = 'test_images/'
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
for img_path in test_images:
    result = predict_image(img_path, model, le_dict)
    print(f"Image: {img_path}")
    print(result)