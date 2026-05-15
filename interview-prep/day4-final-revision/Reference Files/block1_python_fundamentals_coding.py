"""
================================================================================
DAY 4 - BLOCK 1: PYTHON FUNDAMENTALS CODING REVISION
================================================================================
Goal:
Prepare for Python coding questions focused on fundamentals, not heavy DSA.

Interview expectation:
For a Senior SDET round, they may ask easy-to-moderate Python problems to check:

    - list usage
    - dict/hash map usage
    - set usage
    - string handling
    - tuple unpacking
    - loops and conditions
    - functions
    - basic OOP thinking
    - edge-case handling

Strategy:
Do not over-focus on advanced DSA. Lead with clean Python, readable code,
correct edge cases, and explain time/space complexity briefly.
================================================================================
"""


# =============================================================================
# QUICK PYTHON REVISION
# =============================================================================
"""
LIST
----
Use when order matters and duplicates are allowed.

Common operations:

    nums = [1, 2, 3]
    nums.append(4)
    nums.pop()
    nums.sort()
    nums[::-1]
    [x * 2 for x in nums]


DICT
----
Use for key-value lookup, counting, grouping.

Common operations:

    freq = {}
    freq["a"] = freq.get("a", 0) + 1

    for key, value in freq.items():
        print(key, value)


SET
---
Use for uniqueness and membership checks.

Common operations:

    seen = set()
    seen.add("a")
    "a" in seen
    set([1, 2]) & set([2, 3])  # intersection


STRING
------
Strings are immutable.

Common operations:

    text.lower()
    text.strip()
    text.split()
    " ".join(words)
    ch.isalnum()
    text[::-1]


TUPLE
-----
Immutable ordered collection.

Useful for unpacking and dictionary keys:

    point = (10, 20)
    x, y = point


EDGE CASES TO ALWAYS THINK ABOUT
--------------------------------
    - empty list/string
    - single element
    - duplicates
    - negative numbers
    - uppercase/lowercase
    - spaces/special characters
    - None input if interviewer asks
"""


# =============================================================================
# PROBLEM 1: REVERSE A STRING
# =============================================================================
"""
Problem:
    Given a string, return the reversed string.

Concepts:
    string slicing, immutability

Example:
    "hello" -> "olleh"

Solution:
"""


def reverse_string(text: str) -> str:
    return text[::-1]


"""
Alternative:
"""


def reverse_string_loop(text: str) -> str:
    result = []
    for char in text:
        result.insert(0, char)
    return "".join(result)


"""
Interview note:
    Slicing is Pythonic. Loop version shows you understand mechanics.
"""


# =============================================================================
# PROBLEM 2: VALID PALINDROME
# =============================================================================
"""
Problem:
    Check whether a string is a palindrome after ignoring case and non-alphanumeric
    characters.

Concepts:
    string normalization, isalnum, two pointers

Example:
    "A man, a plan, a canal: Panama" -> True
"""


def is_palindrome(text: str) -> bool:
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointers(text: str) -> bool:
    left, right = 0, len(text) - 1

    while left < right:
        while left < right and not text[left].isalnum():
            left += 1
        while left < right and not text[right].isalnum():
            right -= 1

        if text[left].lower() != text[right].lower():
            return False

        left += 1
        right -= 1

    return True


# =============================================================================
# PROBLEM 3: CHARACTER FREQUENCY
# =============================================================================
"""
Problem:
    Count frequency of each character in a string.

Concepts:
    dict, get()

Example:
    "hello" -> {"h": 1, "e": 1, "l": 2, "o": 1}
"""


def char_frequency(text: str) -> dict[str, int]:
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    return freq


# =============================================================================
# PROBLEM 4: FIRST NON-REPEATING CHARACTER
# =============================================================================
"""
Problem:
    Return the first character that appears only once. If none exists, return None.

Concepts:
    dict preserves insertion order in modern Python

Example:
    "swiss" -> "w"
"""


def first_non_repeating_char(text: str) -> str | None:
    freq = char_frequency(text)

    for char in text:
        if freq[char] == 1:
            return char

    return None


# =============================================================================
# PROBLEM 5: VALID ANAGRAM
# =============================================================================
"""
Problem:
    Check whether two strings are anagrams.

Concepts:
    frequency map, sorting alternative

Example:
    "listen", "silent" -> True
"""


def is_anagram(first: str, second: str) -> bool:
    if len(first) != len(second):
        return False
    return char_frequency(first) == char_frequency(second)


def is_anagram_sorting(first: str, second: str) -> bool:
    return sorted(first) == sorted(second)


# =============================================================================
# PROBLEM 6: REMOVE DUPLICATES FROM LIST
# =============================================================================
"""
Problem:
    Remove duplicates while preserving original order.

Concepts:
    set for membership, list for output order

Example:
    [1, 2, 2, 3, 1] -> [1, 2, 3]
"""


def remove_duplicates_preserve_order(items: list[int]) -> list[int]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# =============================================================================
# PROBLEM 7: FIND DUPLICATES
# =============================================================================
"""
Problem:
    Return duplicate values from a list once.

Concepts:
    set, membership

Example:
    [1, 2, 2, 3, 3, 3] -> [2, 3]
"""


def find_duplicates(items: list[int]) -> list[int]:
    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


# =============================================================================
# PROBLEM 8: COMMON ELEMENTS IN TWO LISTS
# =============================================================================
"""
Problem:
    Return common elements between two lists.

Concepts:
    set intersection

Example:
    [1, 2, 3], [2, 3, 4] -> [2, 3]
"""


def common_elements(first: list[int], second: list[int]) -> list[int]:
    return list(set(first) & set(second))


# =============================================================================
# PROBLEM 9: SECOND LARGEST NUMBER
# =============================================================================
"""
Problem:
    Find second largest unique number.

Concepts:
    set, sorting, edge cases

Example:
    [10, 5, 20, 20] -> 10
"""


def second_largest(nums: list[int]) -> int | None:
    unique_nums = sorted(set(nums), reverse=True)
    if len(unique_nums) < 2:
        return None
    return unique_nums[1]


def second_largest_single_pass(nums: list[int]) -> int | None:
    first = second = None

    for num in nums:
        if num == first or num == second:
            continue
        if first is None or num > first:
            second = first
            first = num
        elif second is None or num > second:
            second = num

    return second


# =============================================================================
# PROBLEM 10: MOVE ZEROES
# =============================================================================
"""
Problem:
    Move all zeroes to the end while keeping non-zero order.

Concepts:
    list construction, two pointers

Example:
    [0, 1, 0, 3, 12] -> [1, 3, 12, 0, 0]
"""


def move_zeroes(nums: list[int]) -> list[int]:
    non_zeroes = [num for num in nums if num != 0]
    zero_count = len(nums) - len(non_zeroes)
    return non_zeroes + [0] * zero_count


def move_zeroes_in_place(nums: list[int]) -> None:
    insert_position = 0

    for num in nums:
        if num != 0:
            nums[insert_position] = num
            insert_position += 1

    while insert_position < len(nums):
        nums[insert_position] = 0
        insert_position += 1


# =============================================================================
# PROBLEM 11: MERGE TWO SORTED LISTS
# =============================================================================
"""
Problem:
    Merge two sorted lists into one sorted list.

Concepts:
    two pointers

Example:
    [1, 3, 5], [2, 4, 6] -> [1, 2, 3, 4, 5, 6]
"""


def merge_sorted_lists(first: list[int], second: list[int]) -> list[int]:
    i = j = 0
    result = []

    while i < len(first) and j < len(second):
        if first[i] <= second[j]:
            result.append(first[i])
            i += 1
        else:
            result.append(second[j])
            j += 1

    result.extend(first[i:])
    result.extend(second[j:])
    return result


# =============================================================================
# PROBLEM 12: GROUP WORDS BY FIRST LETTER
# =============================================================================
"""
Problem:
    Group words by their first character.

Concepts:
    dict of lists, setdefault

Example:
    ["apple", "ant", "bat"] -> {"a": ["apple", "ant"], "b": ["bat"]}
"""


def group_by_first_letter(words: list[str]) -> dict[str, list[str]]:
    grouped = {}

    for word in words:
        if not word:
            continue
        key = word[0].lower()
        grouped.setdefault(key, []).append(word)

    return grouped


# =============================================================================
# PROBLEM 13: GROUP ANAGRAMS
# =============================================================================
"""
Problem:
    Group words that are anagrams.

Concepts:
    dict grouping, sorted string as key

Example:
    ["eat", "tea", "tan", "ate"] -> [["eat", "tea", "ate"], ["tan"]]
"""


def group_anagrams(words: list[str]) -> list[list[str]]:
    groups = {}

    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)

    return list(groups.values())


# =============================================================================
# PROBLEM 14: WORD FREQUENCY
# =============================================================================
"""
Problem:
    Count frequency of words in a sentence.

Concepts:
    split, lower, dict

Example:
    "Hello hello world" -> {"hello": 2, "world": 1}
"""


def word_frequency(sentence: str) -> dict[str, int]:
    freq = {}

    for word in sentence.lower().split():
        freq[word] = freq.get(word, 0) + 1

    return freq


# =============================================================================
# PROBLEM 15: FLATTEN NESTED LIST
# =============================================================================
"""
Problem:
    Flatten a nested list of integers.

Concepts:
    recursion, isinstance

Example:
    [1, [2, [3, 4]], 5] -> [1, 2, 3, 4, 5]
"""


def flatten_list(items: list) -> list[int]:
    result = []

    for item in items:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)

    return result


# =============================================================================
# PROBLEM 16: VALID PARENTHESES
# =============================================================================
"""
Problem:
    Check whether brackets are balanced.

Concepts:
    stack, dict mapping

Example:
    "{[]}" -> True
    "{[}]" -> False
"""


def valid_parentheses(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []

    for char in text:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False

    return not stack


# =============================================================================
# PROBLEM 17: SORT LIST OF DICTIONARIES
# =============================================================================
"""
Problem:
    Sort list of dictionaries by a key.

Concepts:
    sorted, lambda

Example:
    [{"name": "A", "age": 30}, {"name": "B", "age": 20}]
    sort by age -> B then A
"""


def sort_users_by_age(users: list[dict]) -> list[dict]:
    return sorted(users, key=lambda user: user["age"])


# =============================================================================
# PROBLEM 18: LIST OF TUPLES TO DICT
# =============================================================================
"""
Problem:
    Convert list of key-value tuples to dict.

Concepts:
    tuple unpacking, dict

Example:
    [("a", 1), ("b", 2)] -> {"a": 1, "b": 2}
"""


def tuples_to_dict(items: list[tuple[str, int]]) -> dict[str, int]:
    result = {}
    for key, value in items:
        result[key] = value
    return result


def tuples_to_dict_builtin(items: list[tuple[str, int]]) -> dict[str, int]:
    return dict(items)


# =============================================================================
# PROBLEM 19: MISSING NUMBER
# =============================================================================
"""
Problem:
    Given numbers from 0 to n with one missing, find the missing number.

Concepts:
    sum formula

Example:
    [3, 0, 1] -> 2
"""


def missing_number(nums: list[int]) -> int:
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# =============================================================================
# PROBLEM 20: RETRY DECORATOR
# =============================================================================
"""
Problem:
    Write a simple retry decorator for flaky operations.

Concepts:
    functions as objects, decorators, exceptions

SDET relevance:
    Retry should be used carefully for unstable infrastructure, not to hide real
    product bugs.
"""


def retry(times: int = 3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None

            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    last_error = error

            raise last_error

        return wrapper

    return decorator


# =============================================================================
# INTERVIEW QUESTIONS AND SHORT ANSWERS
# =============================================================================
"""
1. Why use dict for frequency problems?

    Dict gives O(1) average lookup/update, so counting characters or words is
    efficient and readable.


2. When would you use set?

    For uniqueness, duplicate detection, and fast membership checks.


3. Why are strings immutable?

    A string cannot be modified in place. Operations like lower(), replace(), or
    slicing return a new string.


4. What is list comprehension?

    A concise way to build lists.

        squares = [x * x for x in nums]


5. What is tuple unpacking?

    Assigning tuple/list elements to variables directly.

        name, age = ("Lokender", 30)


6. What is the difference between list and tuple?

    List is mutable; tuple is immutable. Tuple can be used as a dict key if all
    contained values are hashable.


7. What is time complexity of dictionary lookup?

    O(1) average, O(n) worst case due to hash collisions, but average O(1) is
    expected in normal use.


8. How do you explain edge cases in interview?

    Before coding, mention empty input, duplicates, single-element input,
    uppercase/lowercase, and invalid characters depending on problem.
"""


# =============================================================================
# MOST IMPORTANT PYTHON THEORY QUESTIONS
# =============================================================================
"""
1. Is Python dynamically typed or statically typed?

    Python is dynamically typed. You do not need to declare variable types before
    assigning values.

    Example:
        value = 10
        value = "hello"

    The same variable can point to different types at runtime.

    Interview line:
        Python is dynamically typed, but type hints help readability, tooling,
        and maintainability in larger projects.


2. What are type hints?

    Type hints document expected input and return types.

    Example:
        def add(a: int, b: int) -> int:
            return a + b

    Important:
        Type hints are not enforced by Python at runtime by default. They help
        IDEs, linters, and tools like mypy/pyright.


3. What is mutability?

    Mutable objects can be changed after creation.
    Immutable objects cannot be changed after creation.

    Mutable:
        list, dict, set

    Immutable:
        int, float, str, tuple, bool

    Example:
        nums = [1, 2]
        nums.append(3)      # same list changed

        text = "hi"
        text.upper()        # returns new string; original text unchanged


4. Why should we be careful with default mutable arguments?

    Bad:
        def add_item(item, bucket=[]):
            bucket.append(item)
            return bucket

    The same list is reused across calls.

    Good:
        def add_item(item, bucket=None):
            if bucket is None:
                bucket = []
            bucket.append(item)
            return bucket

    Interview line:
        Avoid mutable defaults because they are created once at function
        definition time, not every time the function is called.


5. What is shallow copy vs deep copy?

    Shallow copy:
        Copies only the outer object. Nested objects are still shared.

    Deep copy:
        Copies outer object and nested objects.

    Example:
        import copy

        shallow = copy.copy(data)
        deep = copy.deepcopy(data)

    SDET relevance:
        Important when modifying test data templates. A shallow copy can leak
        nested state between tests.


6. What is the difference between == and is?

    == checks value equality.
    is checks object identity.

    Example:
        a = [1, 2]
        b = [1, 2]

        a == b   # True, same values
        a is b   # False, different objects

    Interview line:
        Use == for comparing values. Use is for None checks:
            if value is None:


7. What is truthy/falsy in Python?

    Falsy values:
        None
        False
        0
        ""
        []
        {}
        set()

    Example:
        if not items:
            print("empty")


8. What is list comprehension?

    A concise way to build a list from an iterable.

    Example:
        even_numbers = [num for num in nums if num % 2 == 0]

    Equivalent loop:
        even_numbers = []
        for num in nums:
            if num % 2 == 0:
                even_numbers.append(num)


9. What is dictionary comprehension?

    A concise way to build a dictionary.

    Example:
        squares = {num: num * num for num in [1, 2, 3]}

    Output:
        {1: 1, 2: 4, 3: 9}


10. What is enumerate?

    enumerate gives index and value while looping.

    Example:
        for index, value in enumerate(["a", "b", "c"]):
            print(index, value)

    Use when:
        You need both position and item.


11. What is zip?

    zip combines multiple iterables element by element.

    Example:
        names = ["A", "B"]
        scores = [90, 80]

        for name, score in zip(names, scores):
            print(name, score)

    Output:
        A 90
        B 80


12. What is *args and **kwargs?

    *args:
        variable number of positional arguments.

    **kwargs:
        variable number of keyword arguments.

    Example:
        def log_event(event_name, *args, **kwargs):
            print(event_name, args, kwargs)

    SDET relevance:
        Useful for wrappers, decorators, custom logging, and retry utilities.


13. What is exception handling?

    try/except handles runtime errors.

    Example:
        try:
            value = int("abc")
        except ValueError:
            value = 0

    Interview warning:
        Do not swallow exceptions silently in tests. Tests should fail clearly.


14. What is finally?

    finally runs whether exception happens or not.

    Use for:
        cleanup
        closing files
        releasing resources


15. What is a decorator?

    A decorator wraps a function to add behavior.

    Example:
        def decorator(func):
            def wrapper(*args, **kwargs):
                print("before")
                result = func(*args, **kwargs)
                print("after")
                return result
            return wrapper

    SDET example:
        retry decorator
        logging decorator
        timing decorator


16. What is lambda?

    A small anonymous function.

    Example:
        users = [{"name": "A", "age": 30}, {"name": "B", "age": 20}]
        sorted_users = sorted(users, key=lambda user: user["age"])


17. What is the difference between sort() and sorted()?

    list.sort():
        modifies the same list in place and returns None.

    sorted():
        returns a new sorted list and works with any iterable.

    Example:
        nums = [3, 1, 2]
        nums.sort()          # nums becomes [1, 2, 3]

        nums = [3, 1, 2]
        new_nums = sorted(nums)


18. What is split and join?

    split:
        string -> list

    join:
        list -> string

    Example:
        words = "hello world".split()
        sentence = " ".join(words)


19. What is isalnum?

    Returns True if a character/string has only letters or numbers.

    Example:
        "a".isalnum()  # True
        "1".isalnum()  # True
        ",".isalnum()  # False

    Used in palindrome questions to ignore punctuation and spaces.


20. What is time complexity?

    Time complexity describes how runtime grows with input size.

    Common examples:
        O(1)     constant
        O(n)     one loop
        O(n^2)   nested loop or repeated list lookup inside loop
        O(n log n) sorting

    Interview line:
        My set/dict lookup solution is usually O(n), while sorting is O(n log n)
        and brute force nested scanning is O(n^2).
"""


# =============================================================================
# PYTHON INTERVIEW MINI DRILLS
# =============================================================================
"""
Drill 1:
    Why use set in duplicate problems?

Answer:
    Set stores unique values and gives O(1) average membership checks.


Drill 2:
    Why use dict in frequency problems?

Answer:
    Dict maps each item to its count and lets me update counts efficiently.


Drill 3:
    Why use list for final output even if set is used?

Answer:
    Set helps with lookup, but list preserves the intended output order.


Drill 4:
    What is the safest way to compare with None?

Answer:
    Use "is None" or "is not None", because None is a singleton.


Drill 5:
    How do you explain a solution?

Answer:
    I explain the data structure, walk through one example, mention edge cases,
    and give time/space complexity.
"""


# =============================================================================
# HOW TO PRACTICE OFFLINE
# =============================================================================
"""
Practice flow:

1. Pick any problem above.
2. Hide the solution.
3. Write solution in notebook or test_python_fundamentals.py.
4. Run with 3 examples:
       normal case
       edge case
       negative/empty case
5. Explain:
       data structure used
       time complexity
       edge cases


Recommended first pass:
    Problems 1-10

Recommended second pass:
    Problems 11-20


Final 30-minute quick revision:
    reverse_string
    is_palindrome
    char_frequency
    first_non_repeating_char
    is_anagram
    remove_duplicates_preserve_order
    second_largest
    group_anagrams
    flatten_list
    valid_parentheses
    retry decorator
"""
