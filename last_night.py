# polynomial 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Flatten
#from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def build_model():
    inputss=Input((1,))
    inputs=Dense(8,activation='relu')(inputss)
    h1=Dense(16,activation='relu')(inputs)
    h2=Dense(32,activation='relu')(h1)
    h3=Dense(16,activation='relu')(h2)
    output=Dense(1,activation='linear')(h3)

    model=Model(inputss,output)
    model.summary()
    return model
    
def polynomial(x):
    return 7*x**4+5*x**3-7*x+10

def data_process():
    n=1000
    x=np.random.randint(0,1000,n).reshape(-1,1)
    x=x/1000.0
    y=np.array([polynomial(i) for i in x])
    y=y.reshape(-1,1)/(1000**4)
    return x,y

def prepare_train_test_val():
    x,y=data_process()
    total_n=len(x)
    train_n=int(total_n*0.7)
    val_n=int(total_n*0.1)
    
    train_x=x[:train_n]
    train_y=y[:train_n]
    
    valx=x[train_n:train_n+val_n]
    valy=y[train_n:train_n+val_n]
    
    test_x=x[train_n+val_n:]
    test_y=y[train_n+val_n:]
    
    return (train_x,train_y),(valx,valy),(test_x,test_y)
    
def showcase(model,test_x,test_y):
    #predict 
    yy=model.predict(test_x)
    #yy=yy*1e8
    
    plt.figure(figsize=(8,5))
    plt.scatter(test_x,test_y,label="original f(x)",alpha=0.6)
    plt.scatter(test_x,yy,label="predict f(x)",alpha=0.6)
    plt.xlabel("Normalized")
    plt.ylabel("Original scale")
    plt.title("Original vs Predicted")
    plt.legend()
    plt.show()

def main():
    model=build_model()
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    
    (train_x,train_y),(valx,valy),(test_x,test_y)=prepare_train_test_val()
    #train
    model.fit(train_x,train_y,validation_data=(valx,valy),epochs=100,batch_size=32)
    #evaluation
    test_loss=model.evaluate(test_x,test_y,batch_size=32)
    loss=test_loss[0]
    print(f"test_loss:{loss:.4f}")
    showcase(model,test_x,test_y)
    
    
    
    
    
if __name__=='__main__':
    main()
    





