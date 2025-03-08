# Codemonk_Machine_Learning_Intern_Assignment
A deep learning model trained on 44.4k images to guess clothing type, gender, season, and color. Built with TensorFlow, served via Flask API and Streamlit app.

The model consisting of:
.eda.py which analyses the given data set, along with outputs in png format which includes categories of clothing graph(category_plot.png) gender divergence graph(gender_plot.png), season based clothing graph(season_plot.png) and sample images(sample_images.png).

.a trained model(train_model.py)that mainly uses styles and images for training and saved as fashion_model.h5 for prediction.

.a test script(test_model.py) to use the model along with a few sample images from amazon to test the model(test_images).

Data set from https://www.kaggle.com/datasets/paramaggarwal/fashion-product- images-dataset. Needs to be downloaded, Extracted and saved as fashion-dataset for analysis and training of the model
Also require fashion_model.h5 for actual trained prediction from the link https://drive.google.com/file/d/1pucriwJXEdGgNde9kZ9bxKoZ9PffKnm-/view?usp=sharing

utilized python and libraries such as
  matplotlib       3.8.0
  numpy            1.26.0
  pandas           2.1.0
  pillow           10.0.0
  scikit-learn     1.3.0
  tensorflow       2.15.0
