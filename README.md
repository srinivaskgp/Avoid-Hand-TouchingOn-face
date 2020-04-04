# Transfer-Learning-in-keras---custom-data 
#in this project downloaded the data of hands of face and face dataset.this 2 classes are trained on  pretrained model imagenet vgg16.
#create data folder and keep the images of face and hand by creating new directories inside it
data	
	face
	hand
#train the model
1 cd Avoid-Hand-TouchingOn-Face --python train.py

#test model on webcame input frames
2.python test.py