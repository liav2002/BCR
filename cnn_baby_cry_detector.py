import numpy as np
from tensorflow import keras
model = keras.models.load_model('output\\model\\cnn')   

def baby_cry_detector(list):
    sum = 0
    for mel_spec in list:
        x = np.reshape(mel_spec, (1,128,431))
        result = model.predict(x,verbose=0)
        if np.argmax(result) == 0:
            sum +=1
    if sum > len(list) / 2.0:
        return True
    else:
        return False


#input - (1,128,431)
#output - (1,4) probability 
