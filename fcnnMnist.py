import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt

def build_model():
    # (train_x,train_y),(test_x,test_y)=mnist.load_data()
    inputs = Input((28,28))
    x=Flatten()(inputs)
    x1=Dense(128,activation='relu')(x)
    x2=Dense(256,activation='relu')(x1)
    x3=Dense(128,activation='relu')(x2)
    outputs=Dense(10,activation='softmax')(x3)
    model =Model(inputs,outputs)
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    return model

def showcase(test_loss,test_accuracy,model_history):
    plt.figure(figsize=(8,5))
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
def main():
    (train_x,train_y),(test_x,test_y)=mnist.load_data()
    train_x=train_x/255.0
    test_x=test_x/255.0
    model =build_model()
    model.summary()
    model_history= model.fit(train_x,train_y,batch_size=32,validation_data=(test_x,test_y),epochs=2)
    
    test_loss,test_accuracy=model.evaluate(test_x,test_y,verbose=0)
    
    print(f"testloss:{test_loss:.4f}")
    
    showcase(test_loss,test_accuracy,model_history)


if __name__=="__main__":
    main()
