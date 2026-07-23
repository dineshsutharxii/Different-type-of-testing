# 1.Explain your framework
# 2.Given a string find the balance paranthesis set
# str_ = '[]{}(){(}'
# 3.Given a string found the sum of integers found
# 4.Given a string found the largest occurrence of consecutive 1’s
# 5. How to handle large datasets for API testing
#
# 1. Find the length of the longest substring without repeating characters
# string: abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
# 2. What are the different status codes in the API.
#
# 1. What is Jenkins and it use, how it is used in you project
# 2. Code question for palindrome and to find the substring for an with without repeating letters
#
# 1.  Have worked with Jenkins, what you have done with it.
# 2.  Diff btw git fetch and git pull
# 3.  What is selenium grid bad explain.
#
# 1.give a detailed explanation about current and previous projects along with tools and technologies stack used.
# 2.Given a bike rental platform with owner and user modules. Write use cases for these scenarios.
# 3.How would you set up a framework for the above ,which tools would you use for the same.
# 4.How would you code the above scenarios in a single test case ,given an idea about the same .
# 5.You need to generate a license . I should have special characters and numbers . Generate license id without using random generator
# 6.Explain the scenario which cannot be automated with selenium and explain how you can automate using API or any other
# 7.Design 1 API test case using framework.
# 8. Given the ecommerce website scenario.
# Asked to design the framework for the same.
# 9. Handling of oauth scenarios in the framework work
# 10. Designing of test cases using framework for one the page in the website
# from selenium.webdriver import ActionChains
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.select import Select
# from selenium.webdriver.support.wait import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC

wait = WebDriverWait(driver, 10)
ele = wait.until(EC.presence_of_element_located(By.ID, ""))

#datatype
int_ = 1
str_ = 'string'
float_ = 5.9
bool_ = True
print(type(int_))
print(type(str_))
print(type(float_))
print(type(bool_))

#collection types
tuple_ = (5, 7, 10, 9, 90, "Hello")  #can't mpdify, No insert/delete
print(tuple_ * 2)
print(tuple_)

list_ = [4, 7, 9, "5"]  #mutable
list_.append("80")
list_.sort(key=lambda x: int(x))
list_.insert(4, 5)
print("Full list : ", list_)
print("Index of '5' is : ", list_.index("5"))
list_.remove(5)
list2_ = ['1', 'g', 60]
list_.extend(list2_)
print(list_)
print(enumerate(list_))

my_dict = {"Name": 'Dipak', "Class": 'V'}  # mutable
print(my_dict["Name"])
my_dict['Marks'] = 98
print(my_dict)
print(my_dict.items())
print(my_dict.values())
print(my_dict.keys())
print(my_dict.pop("Name"))
print(my_dict)

set_ = set([1, 1, 1, 1, 3])  #mutable
set_.add(4)
set_.add(3)
print(set_)


#functions
def nf(): return 5


print(nf())
lambdaf = lambda x: x ** 2
print(lambdaf(5))


#classes and object
class Car:
    def __init__(self, model, brand):
        self.model = model
        self.brand = brand

    def display(self):
        print(f'Car: {self.brand} {self.model}')


car = Car("X4", 'BMW')
car.display()



#Inheritance
class ElectricCar(Car):
    def __init__(self, model, brand, battery):
        super().__init__(model, brand)
        self.battery = battery


ec = ElectricCar("GLE", "Merc", "200kwh")
print(ec.brand, ec.battery)

#Exception handling
try:
    x = 5 / 1
except Exception as e:
    print(e)
else:
    print("No exception")

#file handling
with open("test.txt", "w") as file:
    file.write("Hello world")
with open("test.txt", "r") as file:
    print(file.read())

#selenium
from selenium import webdriver

# from selenium.webdriver.support import expected_conditions as EC

serv_obj = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=serv_obj)

driver.get("https://google.com")
print(driver.title)

web_ele = driver.find_element(By.ID, "p")
web_eles = driver.find_elements(By.XPATH, "//*[@id='p']")

driver.implicitly_wait(5)
wait = WebDriverWait()
wait.until(EC.presence_of_element_located((By.ID, "search")))

act = ActionChains(driver)
act.move_to_element(web_ele).context_click()

driver.switch_to.frame(index, name_frame, frame_id, frame_as_web_element)
driver.switch_to.default_content()

dropdown = Select(driver.find_element(By.ID, "p"))
dropdown.select_by_index(0)
dropdown.select_by_value("value")
dropdown.select_by_visible_text("text")
options = dropdown.options
for option in options:
    print(option.text)
selected_option = dropdown.first_selected_option
print(selected_option.text)
import json
from pprint import pprint

import requests

url = "https://jsonplaceholder.typicode.com/posts"
#get request
get_response = requests.get(url + "/1")
print(get_response.status_code)
# pprint(get_response.json())
print(json.dumps(get_response.json(), indent=4))

#post request
data = {
    "title": "API Testing",
    "body": "Practicing API calls",
    "userId": 5
}
post_res = requests.post(url, json=data)
print(post_res.status_code)
print(post_res.json())

data_put = {
    "title": "API Testing",
    "body": "Practicing API",
    "userId": 2
}
put_res = requests.put(url + "/1", json=data_put)
print(put_res.status_code)
print(json.dumps(put_res.json(), indent=4))

# patch request
partial_update = {"title": "Partially Updated Title"}
patch_res = requests.patch(url + "/1", json=partial_update)
print(patch_res.status_code)
print(json.dumps(patch_res.json(), indent=4))

import os

# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.wait import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager

current_dir = os.getcwd()
folder_path = os.path.join(current_dir, "new_folder")
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

for i in range(3):
    with open(f'{folder_path}/test_{i}.txt', 'w') as file:
        file.write(f"Hello from test_{i}")

all_files = os.listdir(folder_path)
print(all_files)
for file in all_files:
    with open(folder_path + "/" + file, 'r') as f:
        print(f.read())

# from selenium import webdriver
# from selenium.webdriver.support import expected_conditions as EC

# serv_obj = Service(executable_path=ChromeDriverManager().install())
# driver = webdriver.Chrome(service=serv_obj)
# driver.get("")
# wait = WebDriverWait(driver, 10, poll_frequency=5)
# elem = wait.until(EC.presence_of_element_located(By.ID, "id_of_element"))


from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC

serv_obj = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=serv_obj)
