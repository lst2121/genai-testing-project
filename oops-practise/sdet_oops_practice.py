class TestResult:
    total_tests = 0

    def __init__(self, test_name, status, duration):
        self.test_name = test_name
        self.status = status
        self.duration = duration
        TestResult.total_tests += 1
    
    def is_passed(self):
        return self.status == "passed"
    
    def is_failed(self):
        return self.status == "failed"
    
    def get_duration(self):
        return self.duration
    
    def get_test_name(self):
        return self.test_name

def __repr__(self):
    return f"TestResult(test_name={self.test_name}, status={self.status}, duration={self.duration})"

result1 = TestResult("login test", "passed", 2.5)
result2 = TestResult("checkout test", "failed", 4.0)

print(result1.is_passed())
print(result2.is_failed())
print(TestResult.total_tests)
print(result1)
print(result2)
