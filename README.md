# asl-interpreter

Hi there! This is our final project for Digital Image Processing. <br/><br/>
Using webcam data, we track the hands in the image using MediaPipe. Then, we feed the landmarks outputted by MediaPipe through our classification model. The resulting prediction is added to the webcam output in the top-left corner. Additionally, users can type with sign language by holding up a sign and pressing space to capture the prediction. We created a custom dataset of 14000(!!) labelled images to train our model. Our model can predict a total of 28 categories of signs (A-Z, space, and del).<br/><br/>
To begin signing, just run ./mediapipe_solution/app.py. If you wish to rebuild the model using your own dataset, change the 'TRAIN_DIR' field in ./mediapipe_solution/Main.ipynb and run the entire notebook. If you wish to build your own custom dataset, we have included ./photographer.py.
