import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Load the styles file
styles = pd.read_csv('fashion-dataset/styles.csv', on_bad_lines='skip')
print("Styles Info:")
print(styles.info())
print("\nFirst 5 Rows:")
print(styles.head())

# Make a bar chart for categories
plt.figure(figsize=(12, 6))
sns.countplot(data=styles, x='masterCategory')
plt.xticks(rotation=45)
plt.title('Categories')
plt.savefig('category_plot.png')
plt.close()

# Make a bar chart for gender
plt.figure(figsize=(8, 4))
sns.countplot(data=styles, x='gender')
plt.title('Gender')
plt.savefig('gender_plot.png')
plt.close()

# Make a bar chart for seasons
plt.figure(figsize=(8, 4))
sns.countplot(data=styles, x='season')
plt.title('Seasons')
plt.savefig('season_plot.png')
plt.close()

# Show some pictures
image_dir = 'fashion-dataset/images/'
sample_ids = styles['id'].head(5)
plt.figure(figsize=(15, 5))
for i, img_id in enumerate(sample_ids):
    img_path = os.path.join(image_dir, f"{img_id}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"ID: {img_id}")
        plt.axis('off')
plt.savefig('sample_images.png')
plt.close()
print("done")