# duck typing = if it walks like a duck, quacks like a duck, then it is a duck
# if it has the necessary attributes and methods, then it is a duck

class Animal:
    alive = True

class Dog(Animal):
    def speak(self):
        print("Woof!")

class Cat(Animal):
    def speak(self):
        print("Meow!")

class Car:
    alive = False
    def speak(self):
        print("Beep Beep!")

animals = [Dog(), Cat(), Car()]

for animal in animals:
    animal.speak()
    print(animal.alive)