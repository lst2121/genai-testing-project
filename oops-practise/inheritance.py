# Inheritance = allows a class to inherit attributes and methods from another class
# Parent class is the class being inherited from
# Child class is the class that inherits from the parent class
# Child class can access all the attributes and methods of the parent class
# Child class can override the attributes and methods of the parent class
# Child class can add new attributes and methods
# Child class can use the attributes and methods of the parent class

class Animal:
    def __init__(self, name):
        self.name = name
        self.is_alive = True
    
    def eat(self):
        print(f"{self.name} is eating")
    
    def sleep(self):
        print(f"{self.name} is sleeping")

class Dog(Animal):
    def speak(self):
        print(f"{self.name} says Woof!")

class Cat(Animal):
    def speak(self):
        print(f"{self.name} says Meow!")

class Mouse(Animal):
    def speak(self):
        print(f"{self.name} says Squeak!")

dog = Dog("Buddy")
cat = Cat("Whiskers")
mouse = Mouse("Jerry")

print(dog.name)
print(cat.name)
print(mouse.name)

print(dog.is_alive)
print(cat.is_alive)
print(mouse.is_alive)

dog.eat()
cat.sleep()
mouse.eat()

dog.speak()
cat.speak()
mouse.speak()