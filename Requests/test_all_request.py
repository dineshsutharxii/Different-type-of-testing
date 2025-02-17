import pytest
import requests
import time
import random

base_url = "https://reqres.in/"
post_id = 1

@pytest.mark.getReuest
def test_getRequest():
    print(" --- Inside get request ---")
    start_time = time.time()
    get_response = requests.get(url=base_url + 'api/users/2')
    end_time = time.time()
    elapsed_time = end_time - start_time
    response = get_response.json()
    print(response)
    print("Using time method : " + str(elapsed_time))
    print("Using response.elapsed : " + str(get_response.elapsed.total_seconds()))
    assert get_response.elapsed.total_seconds() < 1
    assert get_response.status_code == 200
    # print(response['data']['id'])
    assert response['data']['id'] == 2

@pytest.mark.postRequest
@pytest.mark.runpostandput
def test_postrequest():
    global post_id
    print("--- Inside post Request ---")
    post_header = {
        "Content-Type": 'application/json'
    }
    Name = "Dipak" + str(random.random() * 100)
    Job = "QA Automation" + str(random.random() * 100)
    json_payload = {
        "name": Name,
        "job": Job
    }
    post_response = requests.post(url=base_url + 'api/users', headers=post_header, json=json_payload)
    assert post_response.status_code == 201
    response = post_response.json()
    print(response)
    with open("id.txt", 'w') as file:
        file.write(response['id'])
    assert response['name'] == Name
    assert response['job'] == Job


@pytest.mark.putRequest
@pytest.mark.runpostandput
def test_putRequest():
    print("--- Inside Put Request ---")
    with open("id.txt", 'r') as file:
        post_id = file.read()

    post_header = {
        "Content-Type": 'application/json'
    }
    Name = "Dipak" + str(random.random() * 100)
    Job = "QA Automation" + str(random.random() * 100)
    json_payload = {
        "name": Name,
        "job": Job
    }
    post_response = requests.post(url=base_url + 'api/users/' + str(post_id), headers=post_header, json=json_payload)
    assert post_response.status_code == 201
    response = post_response.json()
    print(response)
    assert response['name'] == Name
    assert response['job'] == Job


@pytest.mark.deleteReuest
def test_deleteRequest():
    print("--- Inside Delete Request ---")
    with open("id.txt", 'r') as file:
        post_id = file.read()
    response = requests.delete(url=base_url + 'api/users/' + str(post_id))
    assert response.status_code == 204

@pytest.mark.skip
def test_get():
    get_req = requests.get(url=base_url + 'api/users/2')
    response = get_req.json()
    assert get_req.status_code == 200
    assert response['data']['id'] == 2
    print(response)


def test_post():
    Name = 'Dipak' + str(random.random() * 100)
    Job = 'HR' + str(random.random() * 100)
    json_load = {
        "name": Name,
        "job": Job
    }
    header_load = {
        "Content-Type": 'application/json'
    }
    post_req = requests.post(url=base_url + 'api/users/', headers=header_load, json=json_load)
    response = post_req.json()
    assert post_req.status_code == 201
    with open("id.txt", 'w') as file:
        file.write(response['id'])
    assert response['name'] == Name
    assert response['job'] == Job