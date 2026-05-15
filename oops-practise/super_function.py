# super() = function used to give access to the methods of a parent class.
# function used in child class to access parent class methods.

class Shape:
    def __init__(self, color, filled):
        self.color = color
        self.filled = filled

    def describe(self):
        print(f"This is a {self.color} and {'filled' if self.filled else 'not filled'}")

class Circle(Shape):
    def __init__(self, color, filled, radius):
        super().__init__(color, filled)
        self.radius = radius
    
    def describe(self):
        super().describe()
        print(f"This is a circle with area {self.radius * self.radius * 3.14}")

class Square(Shape):
    def __init__(self, color, filled, width):
        super().__init__(color, filled)
        self.side = width
    
    def describe(self):
        super().describe()
        print(f"This is a square with area {self.side * self.side}")

class Triangle(Shape):
    def __init__(self, color, filled, width, height):
        super().__init__(color, filled)
        self.width = width
        self.height = height

    def describe(self):
        super().describe()
        print(f"This is a triangle with area {self.width * self.height / 2}")

circle = Circle("red", True, 10)
square = Square("blue", False, 10)
triangle = Triangle("green", True, 10, 10)

print(circle.color)
print(square.color)
print(triangle.color)

print(circle.filled)

print(f"Area of Triangle: {triangle.width * triangle.height / 2}")

circle.describe()
square.describe()
triangle.describe()