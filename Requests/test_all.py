import json
import time

import pytest
import requests
import random

base_url = "https://reqres.in/"
post_id = 1
# token = '66db568fb03cdd1bf01ef3e74d380c282f66ea00e2ff814f63759035b632bb17'

@pytest.mark.getReuest
def test_getRequest():
    print(" --- Inside get request ---")
    header = {}
    payload = {}
    start_time = time.time()
    response = requests.get(url=base_url + 'api/users/2', headers=header, json=payload)
    end_time = time.time()
    res_json = response.json()
    print(res_json)
    assert response.status_code == 200
    assert end_time - start_time < 5

@pytest.mark.postRequest
def test_postRequest():
    print(" --- Inside post request ---")
    header = {
        'Content-Type': 'application/json'
        #, 'Authorization': 'Bearer ' + token
    }
    Name = "Dipak" + str(random.random() * 100)
    Job = "Test" + str(random.random() * 100)
    json_payload = {
        "name": Name,
        "job": Job
    }
    response = requests.post(url=base_url + 'api/users', headers= header, json=json_payload)
    res_json = response.json()
    assert response.status_code == 201
    with open('id.txt', 'w+') as f, open('respone.json', 'w+') as res_file:
        f.write(res_json['id'])
        json.dump(res_json, res_file, indent=4)
    assert res_json['name'] == Name
    assert res_json['job'] == Job

@pytest.mark.putRequest
def test_putRequest():
    print(" --- Inside put request ---")
    header = {
        'Content-Type': 'application/json'
        # , 'Authorization': 'Bearer ' + token
    }
    Name = "Dipak" + str(random.random() * 100)
    Job = "Test" + str(random.random() * 100)
    json_payload = {
        "name": Name,
        "job": Job
    }
    with open("id.txt", "r+") as f, open('respone.json', 'r+') as res_json:
        id = f.read()
        data = json.load(res_json)
    print(data)
    print(json.dumps(data, indent=4))
    response = requests.put(url=base_url + 'api/users/' + str(id), json=json_payload, headers=header)
    res_json = response.json()
    print(json.dumps(res_json, indent=3))
    assert response.status_code == 200
    assert res_json['name'] == Name
    assert res_json['job'] == Job
    assert res_json['name'] != data['name']
    assert res_json['job'] != data['job']

@pytest.mark.deleteReuest
def test_deleteRequest():
    print("--- Inside Delete Request ---")
    with open('id.txt', 'r+') as file:
        id = file.read()
    response = requests.delete(url=base_url + 'api/users/' + str(id))
    assert response.status_code == 204