"""
================================================================================
BLOCK 1: PYTHON WARM-UP (45 min)
================================================================================
HackerRank Python Domain Problems
Refresh Python basics before diving into SDET-specific problems
================================================================================
"""

# =============================================================================
# CONCEPT 1: PRINT & BASIC I/O
# =============================================================================
"""
KEY POINTS:
- print() adds newline by default, customize with end=""
- print(a, b, c) uses space as separator, customize with sep="-"
- input() always returns string, cast with int(), float()
- f-strings are the modern way: f"Hello {name}"
- print(*list) unpacks list as arguments

CODE EXAMPLES:
    print("Hello", "World", sep="-")  # Hello-World
    print("No newline", end="")
    print(*[1,2,3])  # 1 2 3
    name = input("Enter name: ")
    age = int(input("Enter age: "))
"""

# -----------------------------------------------------------------------------
# PROBLEM 1: Say "Hello, World!" (HackerRank)
# -----------------------------------------------------------------------------
"""
Print the string "Hello, World!" to stdout.

This is the classic first program in any language.

Input: None
Output: Hello, World!
"""

def hello_world():
    print("Hello, World!")


# =============================================================================
# CONCEPT 2: CONDITIONALS (if/elif/else)
# =============================================================================
"""
KEY POINTS:
- Python uses elif (not else if or elseif)
- Chained comparisons: 0 < n < 100 (no need for 'and')
- Ternary operator: result = "yes" if condition else "no"
- Truthy/Falsy: None, 0, 0.0, '', [], {}, set() are Falsy
- 'is' checks identity (same object), '==' checks equality (same value)
- Logical operators: and, or, not (NOT &&, ||, !)
- No switch/case before Python 3.10, use match/case in 3.10+

CODE EXAMPLES:
    # Chained comparison
    if 0 < age < 120:
        print("Valid age")
    
    # Ternary
    status = "adult" if age >= 18 else "minor"
    
    # Identity vs Equality
    a = [1, 2]
    b = [1, 2]
    a == b  # True (same values)
    a is b  # False (different objects)
    
    # Match-case (Python 3.10+)
    match status_code:
        case 200: return "OK"
        case 404: return "Not Found"
        case _: return "Unknown"
"""

# -----------------------------------------------------------------------------
# PROBLEM 2: Python If-Else (HackerRank)
# -----------------------------------------------------------------------------
"""
Given an integer n, perform the following conditional actions:
- If n is odd, print "Weird"
- If n is even and in the inclusive range of 2 to 5, print "Not Weird"
- If n is even and in the inclusive range of 6 to 20, print "Weird"
- If n is even and greater than 20, print "Not Weird"

Input: 3
Output: Weird
Explanation: 3 is odd

Input: 24  
Output: Not Weird
Explanation: 24 is even and greater than 20
"""

def weird_or_not(n: int) -> str:
    if n % 2 != 0:
        return "Weird"
    elif 2 <= n <= 5:
        return "Not Weird"
    elif 6 <= n <= 20:
        return "Weird"
    return "Not Weird"


# =============================================================================
# CONCEPT 3: LOOPS (for, while)
# =============================================================================
"""
KEY POINTS:
- range(n): generates 0 to n-1
- range(start, stop): generates start to stop-1
- range(start, stop, step): with custom step
- for-else: else block runs if loop completes without break
- while loop: continues while condition is True
- break: exit loop immediately
- continue: skip to next iteration

CODE EXAMPLES:
    # Basic range
    for i in range(5):      # 0, 1, 2, 3, 4
    for i in range(2, 5):   # 2, 3, 4
    for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    for i in range(5, 0, -1): # 5, 4, 3, 2, 1
    
    # for-else (unique to Python!)
    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                break
        else:
            print(f"{n} is prime")  # runs if no break
    
    # While with else
    while condition:
        process()
    else:
        print("Loop completed normally")
"""

# -----------------------------------------------------------------------------
# PROBLEM 3: Loops (HackerRank)
# -----------------------------------------------------------------------------
"""
The provided code stub reads an integer n from STDIN. 
For all non-negative integers i < n, print i^2 (i squared).

Input: 5
Output:
0
1
4
9
16

Note: Use a loop to iterate from 0 to n-1 and print each square.
"""

def print_squares(n: int) -> None:
    for i in range(n):
        print(i ** 2)


def get_squares(n: int) -> list[int]:
    """Return version for testing"""
    return [i ** 2 for i in range(n)]


# =============================================================================
# CONCEPT 4: PRINT FUNCTION & STRING OPERATIONS
# =============================================================================
"""
KEY POINTS:
- print(*range(1, n+1), sep="") prints without spaces
- "".join(iterable) concatenates strings efficiently
- str.join() is faster than += in loops
- String multiplication: "ab" * 3 = "ababab"
- String slicing: s[start:stop:step]

CODE EXAMPLES:
    # Print without newline between numbers
    print(*range(1, 6), sep="")  # 12345
    
    # Join is more efficient for concatenation
    "".join(str(i) for i in range(1, 6))  # "12345"
    "-".join(["a", "b", "c"])  # "a-b-c"
    
    # String slicing
    s = "hello"
    s[::-1]   # "olleh" (reverse)
    s[1:4]    # "ell"
    s[::2]    # "hlo" (every 2nd char)
"""

# -----------------------------------------------------------------------------
# PROBLEM 4: Print Function (HackerRank)
# -----------------------------------------------------------------------------
"""
Read an integer n. Without using any string methods, print the list of 
integers from 1 to n as a string, without spaces.

Input: 3
Output: 123

Input: 5
Output: 12345

Constraint: Don't use str() explicitly in print - use print's features!
"""

def print_1_to_n(n: int) -> None:
    # Method 1: Using print with sep
    print(*range(1, n + 1), sep="")


def get_1_to_n(n: int) -> str:
    """Return version for testing"""
    return "".join(str(i) for i in range(1, n + 1))


# =============================================================================
# CONCEPT 5: LIST COMPREHENSIONS
# =============================================================================
"""
KEY POINTS:
- Syntax: [expression for item in iterable if condition]
- More Pythonic and often faster than loops
- Can be nested: [x for row in matrix for x in row]
- Dict comprehension: {k: v for k, v in items}
- Set comprehension: {x for x in items}
- Generator expression: (x for x in items) - memory efficient
- Use regular loops for complex logic (readability matters)

CODE EXAMPLES:
    # Basic with condition
    evens = [x for x in range(10) if x % 2 == 0]
    
    # Transform
    upper = [s.upper() for s in words]
    
    # Nested (flatten 2D list)
    flat = [x for row in matrix for x in row]
    
    # Dict comprehension
    squares = {x: x**2 for x in range(5)}
    
    # Conditional expression IN comprehension
    labels = ["even" if x%2==0 else "odd" for x in range(5)]
    
    # Generator (for large data)
    total = sum(x**2 for x in range(1000000))
"""

# -----------------------------------------------------------------------------
# PROBLEM 5: List Comprehensions (HackerRank)
# -----------------------------------------------------------------------------
"""
Given three integers x, y, z and n, print a list of all possible 
coordinates [i, j, k] on a 3D grid where:
- 0 <= i <= x
- 0 <= j <= y  
- 0 <= k <= z
- i + j + k != n

Input: x=1, y=1, z=1, n=2
Output: [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,1]]
Explanation: [0,1,1], [1,0,1], [1,1,0] are excluded because sum = 2

The output should be in lexicographic order (sorted).
"""

def list_comprehension_coords(x: int, y: int, z: int, n: int) -> list[list[int]]:
    return [
        [i, j, k]
        for i in range(x + 1)
        for j in range(y + 1)
        for k in range(z + 1)
        if i + j + k != n
    ]


# =============================================================================
# CONCEPT 6: BUILT-IN FUNCTIONS FOR ITERABLES
# =============================================================================
"""
KEY POINTS:
- enumerate(iterable, start=0): yields (index, value) pairs
- zip(iter1, iter2): parallel iteration, stops at shortest
- sorted(iterable, key=func, reverse=bool): returns sorted list
- reversed(sequence): returns reverse iterator
- any(iterable): True if any element is truthy
- all(iterable): True if all elements are truthy
- sum(), min(), max(): with optional key parameter
- map(func, iterable): apply function to each element
- filter(func, iterable): keep elements where func returns True

CODE EXAMPLES:
    # enumerate
    for idx, val in enumerate(['a', 'b', 'c']):
        print(f"{idx}: {val}")
    
    # zip
    for name, score in zip(names, scores):
        print(f"{name}: {score}")
    
    # sorted with key
    sorted(words, key=len)  # by length
    sorted(words, key=str.lower)  # case-insensitive
    
    # any/all
    has_negative = any(x < 0 for x in nums)
    all_positive = all(x > 0 for x in nums)
    
    # min/max with key
    longest = max(words, key=len)
"""

# -----------------------------------------------------------------------------
# PROBLEM 6: Find the Runner-Up Score (HackerRank)
# -----------------------------------------------------------------------------
"""
Given the participants' score sheet for your University Sports Day, 
find the runner-up score (second highest).

You are given n scores. Find the second largest score.

Input: n=5, scores=[2, 3, 6, 6, 5]
Output: 5
Explanation: Maximum is 6, runner-up (second max) is 5

Note: There can be duplicate scores. The runner-up is the second 
DISTINCT highest score.

Constraints:
- 2 <= n <= 10
- -100 <= scores[i] <= 100
"""

def runner_up_score(scores: list[int]) -> int:
    """Using set and sorted - O(n log n)"""
    unique = sorted(set(scores), reverse=True)
    return unique[1]


def runner_up_score_optimal(scores: list[int]) -> int:
    """Single pass approach - O(n)"""
    first = second = float('-inf')
    
    for score in scores:
        if score > first:
            second = first
            first = score
        elif score > second and score != first:
            second = score
    
    return second


# =============================================================================
# BONUS: NESTED LISTS (HackerRank)
# =============================================================================
"""
Given names and grades of N students, find students with second lowest grade.

Input:
5
Harry 37.21
Berry 37.21
Tina 37.2
Akriti 41
Harsh 39

Output:
Berry
Harry

Explanation: Tina has lowest (37.2), Harry and Berry have second lowest (37.21)
Output names alphabetically.
"""

def second_lowest_students(records: list[tuple[str, float]]) -> list[str]:
    grades = sorted(set(grade for name, grade in records))
    second_lowest = grades[1]
    
    students = sorted(name for name, grade in records if grade == second_lowest)
    return students


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 1: PYTHON WARM-UP - TEST CASES")
    print("=" * 60)
    
    # Problem 1: Hello World
    print("\n1. Hello World:")
    hello_world()
    
    # Problem 2: Weird or Not
    print("\n2. Weird or Not:")
    assert weird_or_not(3) == "Weird", "3 is odd"
    assert weird_or_not(4) == "Not Weird", "4 is even, 2-5"
    assert weird_or_not(12) == "Weird", "12 is even, 6-20"
    assert weird_or_not(24) == "Not Weird", "24 is even, >20"
    print("   All test cases passed!")
    
    # Problem 3: Loops
    print("\n3. Loops (Squares):")
    assert get_squares(5) == [0, 1, 4, 9, 16]
    print("   All test cases passed!")
    
    # Problem 4: Print Function
    print("\n4. Print Function:")
    assert get_1_to_n(5) == "12345"
    assert get_1_to_n(3) == "123"
    print("   All test cases passed!")
    
    # Problem 5: List Comprehensions
    print("\n5. List Comprehensions:")
    result = list_comprehension_coords(1, 1, 1, 2)
    expected = [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,1]]
    assert result == expected
    print("   All test cases passed!")
    
    # Problem 6: Runner-Up Score
    print("\n6. Runner-Up Score:")
    assert runner_up_score([2, 3, 6, 6, 5]) == 5
    assert runner_up_score_optimal([2, 3, 6, 6, 5]) == 5
    assert runner_up_score([5, 5, 5, 4, 4, 3]) == 4
    print("   All test cases passed!")
    
    # Bonus: Second Lowest Students
    print("\n7. Bonus - Second Lowest Students:")
    records = [("Harry", 37.21), ("Berry", 37.21), ("Tina", 37.2), 
               ("Akriti", 41), ("Harsh", 39)]
    assert second_lowest_students(records) == ["Berry", "Harry"]
    print("   All test cases passed!")
    
    print("\n" + "=" * 60)
    print("BLOCK 1 COMPLETE!")
    print("=" * 60)
