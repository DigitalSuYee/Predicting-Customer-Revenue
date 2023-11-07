import requests

url = 'http://localhost:9696/predict'

user_id = 'xyz-123'

user = {
'administrative': 0,
'administrative_duration': 0.0,
'informational': 0,
'informational_duration': 0.0,
'productrelated': 1,
'productrelated_duration': 0.000000,
'bouncerates': 0.200000,
'exitrates': 0.200000,
'pagevalues': 0.000000,
'specialday': 0.0,
'month': 'Feb',
'operatingsystems': 1,
'browser': 1,
'region': 1,
'trafiictype': 1,
'visitortype': 'Returning_Visitor',
'weekend': 'False'
}

response = requests.post(url, json=user).json()
print(response)

if response['revenue'] == True:
    print('Recommend personalized product %s' % user_id)
else:
    print('Not recommend personalized product %s' % user_id)