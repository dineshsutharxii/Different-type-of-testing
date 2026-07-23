#datatype
_int = 0
_string = 'string'
_float = 1.0
_boolean = True

print(type(_int))
print(type(_float))
print(type(_string))

#collection
listy = [1, 3, 5, 'abc', False]
print(type(listy))
listy.append('xyz')
listy.extend([1, 4, 6])
listy.reverse()
listy.remove(1)
print(listy)

_tuple = (1, 2, 5, [3, 6, 8, 'abc'], 'abc')
print(_tuple[3][1])

dict1 = {'one': 1, 'two': 2, 'name': 'dipak', 'age': 20}
dict1['add'] = 'NA'
print(dict1)
print(dict1.values())
print(dict1.keys())

_set = {1, 2, 3, 2, 5}
_set.add(5)
_set.add(6)
print(_set)
print(set(_set))

#loops
for i in range(5):
    print(i)
for ele in listy:
    print(ele)

#conditional
age = 25
if age > 18:
    print("Adult")
elif age == 18:
    print("Exactly 18")
else:
    print("Minor")


#functions
def addition(a, b):
    return a + b


print(addition(5, 6))
square = lambda x: x * x
print(square(5))

#List Comprehension & Built-ins
square = [x * x for x in range(1, 5)]
print(square)
even = [x for x in range(5) if x % 2 == 0]
print(even)

print(sum(even))
print(max(even))
print(min(even))
print(len(even))


##oops
#class and object
class Car:
    company = "Honda"  #class variable

    def __init__(self, name):
        self.name = name  # instance variable

    def display(self):  #instance method
        print(self.name)

    @classmethod
    def change_name(cls, name):  #class methods
        cls.company = name

    @staticmethod
    def add_two_number(a, b):  #static method
        return a + b


car = Car('Civic')
car.display()
print(Car.company)
car.change_name('BMW')
print(car.company)
print(Car.add_two_number(1, 4))
print(car.add_two_number(4, 2))


#encapsulation
# __var - private variable : A public variable can be accessed from anywhere.
# _var - protected variable : Protected variables are meant to be used inside the class and its subclasses.
# var - public variable : It appears inaccessible from outside the class. Double underscore triggers name mangling.
#Python internally changes the variable name.
#__svar becomes _classnaem__var  This is called name mangling.
class Bank:
    def __init__(self):
        self.__balance = 1000

    def get_balance(self):
        return self.__balance


bank = Bank()
print(bank.get_balance())


class Employee:

    def __init__(self):
        self.__salary = 50000


emp = Employee()
print(emp.__dict__)
print(emp._Employee__salary)  #mangling effect


#Inheritance
class Animal:
    def sound(self):
        print("Animal")


class Dog(Animal):
    def streetdog(self):
        print("Street Dog")


class Area(Dog):
    def streetdog(self):
        super().streetdog()
        print("Bangalore")


dog = Dog()
dog.sound()

area = Area()
area.streetdog()
