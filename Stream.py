import numpy as np
import pickle 
load_scaler = pickle.load(open(r'C:\Users\HP\Desktop\pcos\pcos_scaler.pkl', 'rb'))
loaded_model = pickle.load(open(r'C:\Users\HP\Desktop\pcos\pcos_model.sav', 'rb'))

input_data = (