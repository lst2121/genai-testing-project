# polymorphism = many forms of the same function
# polymorphism = ability to take different forms

# Two ways to achieve polymorphism:
# 1. Duck typing = if it walks like a duck, quacks like a duck, then it is a duck (object must have necessary attributes and methods)
# 2. Inheritance = if a class is a subclass of another class, then it is a subclass (an object could be treated of same type as a aprent class)
from abc import ABC, abstractmethod

class Shape:

    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

class Pizza(Circle):
    def __init__(self, topping, radius):
        super().__init__(radius)
        self.topping = topping

shapes = [Circle(4), Square(5), Triangle(6, 7), Pizza("pepperoni", 8)]

for shape in shapes:
    print(shape.area())
