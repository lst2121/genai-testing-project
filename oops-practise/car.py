class Car:
    def __init__(self, model, year, color, for_sale):
        self.model = model
        self.year = year
        self.color = color
        self.for_sale = for_sale
    
    def drive(self):
        print(f"Driving {self.model} {self.year} {self.color}")
    
    def stop(self):
        print(f"Stopping {self.model} {self.year} {self.color}")
    
    def describe(self):
        print(f"This is a {self.model} {self.year} {self.color} car")