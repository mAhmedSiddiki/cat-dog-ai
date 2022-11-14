from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model =load_model("model/cat_dog.h5")

def detect_cat_dog(pet):
  test_image = load_img(pet, target_size = (200, 200))
  
  test_image = img_to_array(test_image)/255
  test_image = np.expand_dims(test_image, axis = 0)
  
  result = model.predict(test_image).round(3)
  print('Result = ', result)

  pred = np.argmax(result)

  if pred == 0:
    return 'Cat'
  else:
    return 'Dog'

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # set input
        filename = file.filename
        print("Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("Predicting class......")
        detect = detect_cat_dog(pet=file_path)
        
        return render_template('predict_cat_dog.html', pred_output = detect, user_image = file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, debug=False)
