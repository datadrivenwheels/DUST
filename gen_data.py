import pickle
import numpy as np



data = np.random.randn(30,3,77,224,224)
with open('./data/video_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,77,224,224)
with open('./data/video_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,77,224,224)
with open('./data/video_test.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(30,3,51)
with open('./data/kine_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,51)
with open('./data/kine_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randn(10,3,51)
with open('./data/kine_test.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randint(0, 2, size=30)
with open('./data/label_train.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randint(0, 2, size=10)
with open('./data/label_val.pkl', 'wb') as file:
    pickle.dump(data, file)
    
data = np.random.randint(0, 2, size=10)
with open('./data/label_test.pkl', 'wb') as file:
    pickle.dump(data, file)