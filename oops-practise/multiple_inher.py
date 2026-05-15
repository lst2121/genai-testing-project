# multiple inheritance = a class that inherits from multiple parent classes
# C(A, B)
# Child class inherits from multiple parent classes

# multi level inheritance is not recommended because it can lead to complex code and conflicts
# C(B) <- B(A) <- A(object)

class Animal:

    def __init__(self, name):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating")
    
    def sleep(self):
        print(f"{self.name} is sleeping")

class Prey(Animal):
    def flee(self):
        print(f"{self.name} flees")

class Predator(Animal):
    def hunt(self):
        print(f"{self.name} hunts.")

class Rabbit(Prey):
    pass

class Hawk(Predator):
    pass

class Fish(Prey, Predator):
    pass

rabbit = Rabbit("Bugs")
hawk = Hawk("Eagle")
fish = Fish("Salmon")

rabbit.flee()
# rabbit.hunt()
hawk.hunt()
# hawk.flee()
fish.flee()
fish.hunt()

rabbit.eat()
hawk.eat()
fish.sleep()