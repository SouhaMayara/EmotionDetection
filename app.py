from flask import Flask, render_template, request
import cv2
import numpy as np 
from keras.models import load_model

app = Flask(__name__) #create the app flask

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #it should load new image every time

@app.route('/') #created a route for index.html
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST']) #created a route for predict.html
def predict():
	image = request.files['select_file']

	image.save('static/file.jpg')

	image = cv2.imread('static/file.jpg')

	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #create a cascade 
	
	faces = cascade.detectMultiScale(gray, 1.1, 3) #detect faces

	for x,y,w,h in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

		cropped = image[y:y+h, x:x+w]  #to save a cropped image


	cv2.imwrite('static/after.jpg', image)
	
	try:
		cv2.imwrite('static/cropped.jpg', cropped)

	except:
		pass #in some cases it will not detect face ==>no cropped variable



	try:
		img = cv2.imread('static/cropped.jpg', 0)

	except:
		img = cv2.imread('static/file.jpg', 0)

	img = cv2.resize(img, (48,48))#convert to 48 by 48 pixel
	img = img/255.0

	img = img.reshape(1,48,48,1) #reshape the array to 1 by size 48 48

	model = load_model('model.h5') #load the model

	pred = model.predict(img) #store the prediction


	label_map = ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']
	#the prediction consist of 6 values in our list so we have to take maximum value 
	pred = np.argmax(pred) 
	
	m=model.predict(img)
	m1=max(m)
	final_pred = label_map[pred]


	return render_template('predict.html', data=final_pred,m=m1[pred],p=pred)


if __name__ == "__main__":
	app.run(debug=True)