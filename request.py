import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Sqft Area':1100, 'Bathrooms':2, 'BHK':2, 'Location':'Indira Nagar'})

print(r.json())