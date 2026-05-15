from abc import ABC, abstractmethod

class Validator(ABC):
    @abstractmethod
    def validate(self, value):
        pass

class EmailValidator(Validator):
    def validate(self, value):
        return "@" in value and "." in value.split("@")[-1]

class StatusValidator(Validator):
    def validate(self, value):
        return value in ["active", "inactive"]

validators = [EmailValidator(), StatusValidator()]

values = ["test@example.com", "inactive"]

for validator in validators:
    for value in values:
        print(validator.validate(value))
