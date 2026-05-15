class Student:

    class_year = 2026
    num_students = 0

    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
        Student.num_students += 1

student1 = Student("John", 20, "A")
student2 = Student("Jane", 21, "B")
print(Student.num_students) # **class variable is shared by all instances**
student3 = Student("Jim", 22, "C")
print(Student.num_students)

print(f"My graduating class of {Student.class_year} has {Student.num_students} students")
print(student1.name)
print(student2.name)
print(student3.name)