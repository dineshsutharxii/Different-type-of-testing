import json
import pytest
import requests
import time

url_jsonpath = "https://jsonplaceholder.typicode.com/posts"
url_gorest = "https://gorest.co.in/public/v2/users"

response = requests.get(url_gorest)
json_response = response.json()
with open("respone.json", "w+") as res:
    json.dump(json_response, res, indent=4)
with open("respone.json", "r+") as res:
    data = json.load(res)
print(data)
print(json.dumps(data, indent=4)) # pretty format