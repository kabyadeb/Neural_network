import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Input,MaxPooling2D,Flatten

def build_model():
    inputs=Input((28,28,1))
    x1=Conv2D(32,(3,3),activation='relu')(inputs)
    x2=MaxPooling2D((2,2))(x1)
    x3=Flatten()(x2)
    x4=Dense(16,activation='relu')(x3)
    model=Model(inputs,x4)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def main():
    (train_x,train_y),(test_x,test_y)=mnist.load_data()
    train_x=train_x.reshape(-1,28,28,1)
    test_x=test_x.reshape(-1,28,28,1)
    
    model=build_model()
    model.summary()
    meo = model.fit(x=train_x, y=train_y, batch_size=32, epochs=10, verbose='auto',validation_data=(test_x,test_y))
    model_loss,model_acc=model.evaluate(x=train_x, y=train_y)
    print(f"Loss is :{model_loss%100}")
    
    plt.figure(figsize=(10,5))
    plt.plot(meo.history['accuracy'],color='green')
    plt.xlabel("epochs")
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.show()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(train_x[i],cmap='gray')
        plt.title(f"label:{train_y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
    
if __name__=='__main__':
    main()
