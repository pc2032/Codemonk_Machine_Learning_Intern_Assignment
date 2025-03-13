# Codemonk_Machine_Learning_Intern_Assignment
A deep learning model trained on 44.4k images to guess clothing type, gender, season, and color. Built with TensorFlow, served via Flask API and Streamlit app.

The model consists of:
eda.py which analyses the given data set, along with outputs in png format which includes categories of clothing graph(category_plot.png) gender divergence graph(gender_plot.png), season based clothing graph(season_plot.png) and sample images(sample_images.png).

A trained model(train_model.py)that mainly uses styles and images for training and saved as fashion_model.h5 for prediction.

A test script(test_model.py) to use the model along with a few sample images from amazon to test the model(test_images).

Data set from https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset. Needs to be downloaded, Extracted and saved as fashion-dataset for analysis and training of the model
Also require fashion_model.h5 for actual trained prediction from the link https://drive.google.com/file/d/1pucriwJXEdGgNde9kZ9bxKoZ9PffKnm-/view?usp=sharing

utilized python and libraries such as
  matplotlib
  numpy
  pandas
  pillow         
  scikit-learn
  tensorflow
