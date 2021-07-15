from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
import matplotlib.pyplot as plt
import matplotlib.image

#getting our data set and split into training ana testing set
(x_train,y_train) , (x_test,y_test) = mnist.load_data()
#normalizing our data
x_train = keras.utils.normalize(x_train)
x_test = keras.utils.normalize(x_test)
#ploting one data
plt.imshow(x_train[1],cmap = plt.cm.binary)


classifier = Sequential()
classifier.add(Flatten())
#adding input layer and first hidden layer 
classifier.add(Dense(units =120, activation = 'relu', kernel_initializer= 'uniform',input_dim=28))
#adding second hideen layer
classifier.add(Dense(units=120,activation = 'relu',kernel_initializer ='uniform'))
#adding output layer
classifier.add(Dense(units = 10,activation = 'softmax'))

classifier.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

classifier.fit(x_train,y_train, epochs = 2)
#evaluating ANN on testing data
val_loss,val_acc = classifier.evaluate(x_test,y_test)
print(val_loss,val_acc) 