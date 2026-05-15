"""
================================================================================
BLOCK 2: SDET PYTHON INTERVIEW QUESTIONS (1.5 hrs)
================================================================================
10 Must-Solve Problems commonly asked in SDET/QA interviews
These test string manipulation, array operations, and problem-solving skills
================================================================================
"""

# =============================================================================
# CONCEPT 1: STRING SLICING & REVERSAL
# =============================================================================
"""
KEY POINTS:
- Strings are immutable in Python (can't modify in place)
- Slicing: s[start:stop:step]
- Reverse: s[::-1] (step of -1)
- Negative indices: s[-1] is last char, s[-2] is second last
- String methods: .lower(), .upper(), .strip(), .split(), .join()
- ord(char) gives ASCII value, chr(num) gives character

CODE EXAMPLES:
    s = "hello"
    s[::-1]      # "olleh" (reverse)
    s[1:4]       # "ell" (slice)
    s[-1]        # "o" (last char)
    s[::2]       # "hlo" (every 2nd char)
    
    # Check if alphanumeric
    char.isalnum()  # True for a-z, A-Z, 0-9
    char.isalpha()  # True for a-z, A-Z only
    char.isdigit()  # True for 0-9 only
"""

# -----------------------------------------------------------------------------
# PROBLEM 1: Reverse String (LeetCode #344)
# -----------------------------------------------------------------------------
"""
Write a function that reverses a string. The input string is given as 
an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]

For SDET: Also know the simple string reversal s[::-1]
"""

def reverse_string_inplace(s: list[str]) -> None:
    """Two pointer approach - O(n) time, O(1) space"""
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


def reverse_string_simple(s: str) -> str:
    """Simple Pythonic way"""
    return s[::-1]


# =============================================================================
# CONCEPT 2: PALINDROME CHECKING
# =============================================================================
"""
KEY POINTS:
- Palindrome reads same forwards and backwards
- Clean the string: remove non-alphanumeric, lowercase
- Two approaches: compare reversed OR two pointers
- Two pointers is more memory efficient O(1) vs O(n)
- Generator expressions save memory: ''.join(c for c in s if c.isalnum())

CODE EXAMPLES:
    # Clean string
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    # Two pointers
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
"""

# -----------------------------------------------------------------------------
# PROBLEM 2: Valid Palindrome (LeetCode #125)
# -----------------------------------------------------------------------------
"""
A phrase is a palindrome if, after converting all uppercase letters to 
lowercase and removing all non-alphanumeric characters, it reads the 
same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Input: s = " "
Output: true
Explanation: After removing non-alphanumeric, s is empty "". 
             Empty string is palindrome.
"""

def is_palindrome_simple(s: str) -> bool:
    """Clean and compare - O(n) time, O(n) space"""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointer(s: str) -> bool:
    """Two pointer approach - O(n) time, O(1) space"""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


# =============================================================================
# CONCEPT 3: ANAGRAMS
# =============================================================================
"""
KEY POINTS:
- Anagram: words with same characters rearranged
- Approach 1: Sort both strings and compare - O(n log n)
- Approach 2: Count characters with Counter - O(n)
- Counter is a dict subclass for counting hashable objects
- Counter supports subtraction: Counter(a) - Counter(b)

CODE EXAMPLES:
    from collections import Counter
    
    # Count characters
    Counter("hello")  # {'l': 2, 'h': 1, 'e': 1, 'o': 1}
    
    # Compare counts
    Counter("listen") == Counter("silent")  # True
    
    # Sort approach
    sorted("listen") == sorted("silent")  # True
"""

# -----------------------------------------------------------------------------
# PROBLEM 3: Valid Anagram (LeetCode #242)
# -----------------------------------------------------------------------------
"""
Given two strings s and t, return true if t is an anagram of s, 
and false otherwise.

An anagram uses all original letters exactly once.

Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false

Follow-up: What if inputs contain Unicode characters?
Answer: Counter works with Unicode, sorting works too.
"""

def is_anagram_sort(s: str, t: str) -> bool:
    """Sort and compare - O(n log n)"""
    return sorted(s) == sorted(t)


def is_anagram_counter(s: str, t: str) -> bool:
    """Counter approach - O(n)"""
    from collections import Counter
    return Counter(s) == Counter(t)


def is_anagram_manual(s: str, t: str) -> bool:
    """Manual counting without Counter - O(n)"""
    if len(s) != len(t):
        return False
    
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    for char in t:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] < 0:
            return False
    
    return True


# =============================================================================
# CONCEPT 4: REMOVING DUPLICATES
# =============================================================================
"""
KEY POINTS:
- set() removes duplicates but loses order
- dict.fromkeys() preserves order (Python 3.7+)
- list(dict.fromkeys(items)) - ordered unique elements
- For sorted arrays: two pointer technique
- seen = set() pattern for tracking

CODE EXAMPLES:
    # Remove duplicates (unordered)
    unique = list(set(items))
    
    # Remove duplicates (preserve order)
    unique = list(dict.fromkeys(items))
    
    # Using seen set
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
"""

# -----------------------------------------------------------------------------
# PROBLEM 4: Contains Duplicate (LeetCode #217)
# -----------------------------------------------------------------------------
"""
Given an integer array nums, return true if any value appears at 
least twice in the array, and return false if every element is distinct.

Input: nums = [1,2,3,1]
Output: true

Input: nums = [1,2,3,4]
Output: false

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
"""

def contains_duplicate(nums: list[int]) -> bool:
    """Using set - O(n) time, O(n) space"""
    return len(nums) != len(set(nums))


def contains_duplicate_early_exit(nums: list[int]) -> bool:
    """Early exit on finding duplicate - O(n) average"""
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


def remove_duplicates_preserve_order(items: list) -> list:
    """Remove duplicates while preserving order"""
    return list(dict.fromkeys(items))


# =============================================================================
# CONCEPT 5: FLATTEN NESTED STRUCTURES
# =============================================================================
"""
KEY POINTS:
- Recursion is natural for nested structures
- isinstance(item, list) checks if item is a list
- itertools.chain.from_iterable() for one level flatten
- Generator functions (yield) for memory efficiency
- Base case + recursive case pattern

CODE EXAMPLES:
    # One level flatten
    from itertools import chain
    list(chain.from_iterable([[1,2], [3,4]]))  # [1,2,3,4]
    
    # List comprehension (one level)
    [x for sublist in nested for x in sublist]
    
    # Recursive flatten (any depth)
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
"""

# -----------------------------------------------------------------------------
# PROBLEM 5: Flatten Nested List (Common Interview Question)
# -----------------------------------------------------------------------------
"""
Given a nested list, flatten it to a single list containing all elements.

Input: [1, [2, 3], [4, [5, 6]], 7]
Output: [1, 2, 3, 4, 5, 6, 7]

Input: [[1, 2], [3, [4, [5]]]]
Output: [1, 2, 3, 4, 5]

This tests recursion understanding - common in SDET interviews.
"""

def flatten_recursive(nested: list) -> list:
    """Recursive approach"""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_recursive(item))
        else:
            result.append(item)
    return result


def flatten_generator(nested: list):
    """Generator approach - memory efficient"""
    for item in nested:
        if isinstance(item, list):
            yield from flatten_generator(item)
        else:
            yield item


def flatten_iterative(nested: list) -> list:
    """Iterative with stack"""
    stack = list(reversed(nested))
    result = []
    
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            result.append(item)
    
    return result


# =============================================================================
# CONCEPT 6: FINDING EXTREMES (MAX, MIN, SECOND LARGEST)
# =============================================================================
"""
KEY POINTS:
- max(iterable), min(iterable) - O(n)
- sorted(iterable)[-2] for second largest - O(n log n)
- Single pass tracking first and second - O(n)
- heapq.nlargest(k, iterable) for k largest - O(n log k)
- float('inf') and float('-inf') for initialization

CODE EXAMPLES:
    # Second largest with sorting
    sorted(set(nums))[-2]
    
    # Using heapq
    import heapq
    heapq.nlargest(2, nums)[-1]  # second largest
    
    # Initialize with infinity
    first = second = float('-inf')
"""

# -----------------------------------------------------------------------------
# PROBLEM 6: Second Largest Element (Common Interview Question)
# -----------------------------------------------------------------------------
"""
Find the second largest element in an array. If duplicates exist,
find the second DISTINCT largest.

Input: [12, 35, 1, 10, 34, 1]
Output: 34
Explanation: Largest is 35, second largest is 34

Input: [10, 10, 10]
Output: None (or raise exception)
Explanation: No second distinct element

Input: [5, 5, 4, 2]
Output: 4
Explanation: Largest is 5, second distinct largest is 4
"""

def second_largest_sort(nums: list[int]) -> int | None:
    """Using sort - O(n log n)"""
    unique = sorted(set(nums), reverse=True)
    return unique[1] if len(unique) >= 2 else None


def second_largest_optimal(nums: list[int]) -> int | None:
    """Single pass - O(n)"""
    if len(nums) < 2:
        return None
    
    first = second = float('-inf')
    
    for num in nums:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num
    
    return second if second != float('-inf') else None


# =============================================================================
# CONCEPT 7: COUNTING & FREQUENCY
# =============================================================================
"""
KEY POINTS:
- collections.Counter - purpose-built for counting
- Counter.most_common(n) - n most frequent elements
- Counter supports arithmetic: +, -, &, |
- defaultdict(int) - alternative for counting
- dict.get(key, default) - safe access with default

CODE EXAMPLES:
    from collections import Counter
    
    # Count words
    words = ["apple", "banana", "apple"]
    Counter(words)  # {'apple': 2, 'banana': 1}
    
    # Most common
    Counter(words).most_common(1)  # [('apple', 2)]
    
    # Manual counting
    count = {}
    for word in words:
        count[word] = count.get(word, 0) + 1
"""

# -----------------------------------------------------------------------------
# PROBLEM 7: Word Frequency / Top K Frequent (LeetCode #347)
# -----------------------------------------------------------------------------
"""
Given an integer array nums and an integer k, return the k most 
frequent elements. You may return the answer in any order.

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Explanation: 1 appears 3 times, 2 appears 2 times

Input: nums = [1], k = 1
Output: [1]

Follow-up: Your algorithm's time complexity must be better than 
O(n log n), where n is the array's size.
"""

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Using Counter - O(n log n) due to sorting"""
    from collections import Counter
    count = Counter(nums)
    return [item for item, freq in count.most_common(k)]


def top_k_frequent_bucket(nums: list[int], k: int) -> list[int]:
    """Bucket sort approach - O(n)"""
    from collections import Counter
    
    count = Counter(nums)
    
    # Bucket: index = frequency, value = list of nums with that frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # Collect from highest frequency buckets
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result


# =============================================================================
# CONCEPT 8: MERGING SORTED ARRAYS
# =============================================================================
"""
KEY POINTS:
- Two pointer technique - each pointer for one array
- Compare elements, take smaller, advance that pointer
- Handle remaining elements when one array exhausts
- In-place merge: start from end to avoid overwriting

CODE EXAMPLES:
    # Merge pattern
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    # Add remaining
    result.extend(a[i:])
    result.extend(b[j:])
"""

# -----------------------------------------------------------------------------
# PROBLEM 8: Merge Sorted Array (LeetCode #88)
# -----------------------------------------------------------------------------
"""
You are given two integer arrays nums1 and nums2, sorted in non-decreasing 
order, and two integers m and n, representing the number of elements in 
nums1 and nums2 respectively.

Merge nums2 into nums1 as one sorted array.

nums1 has length m + n (extra space for nums2 elements).

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
"""

def merge_sorted_inplace(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """Merge in-place from the end - O(m+n) time, O(1) space"""
    # Start from the end to avoid overwriting
    p1 = m - 1  # Last element of nums1's actual values
    p2 = n - 1  # Last element of nums2
    p = m + n - 1  # Last position in nums1
    
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # Copy remaining nums2 elements (if any)
    nums1[:p2 + 1] = nums2[:p2 + 1]


def merge_sorted_new_array(arr1: list[int], arr2: list[int]) -> list[int]:
    """Merge into new array - O(m+n) time and space"""
    result = []
    i = j = 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    
    return result


# =============================================================================
# CONCEPT 9: MISSING NUMBER PROBLEMS
# =============================================================================
"""
KEY POINTS:
- Mathematical approach: expected_sum - actual_sum
- Sum of 1 to n: n * (n + 1) // 2
- XOR approach: a ^ a = 0, a ^ 0 = a
- Set difference approach: set(expected) - set(actual)

CODE EXAMPLES:
    # Sum formula
    expected = n * (n + 1) // 2
    missing = expected - sum(nums)
    
    # XOR all indices and values
    result = 0
    for i in range(n + 1):
        result ^= i
    for num in nums:
        result ^= num
    # result is the missing number
"""

# -----------------------------------------------------------------------------
# PROBLEM 9: Missing Number (LeetCode #268)
# -----------------------------------------------------------------------------
"""
Given an array nums containing n distinct numbers in the range [0, n], 
return the only number in the range that is missing from the array.

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, range is [0,3].
             2 is missing from [0,1,3].

Input: nums = [0,1]
Output: 2

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8

Follow-up: Could you implement a solution using only O(1) extra 
space complexity and O(n) runtime complexity?
"""

def missing_number_sum(nums: list[int]) -> int:
    """Sum formula - O(n) time, O(1) space"""
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


def missing_number_xor(nums: list[int]) -> int:
    """XOR approach - O(n) time, O(1) space"""
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result


def missing_number_set(nums: list[int]) -> int:
    """Set approach - O(n) time, O(n) space"""
    full_set = set(range(len(nums) + 1))
    return (full_set - set(nums)).pop()


# =============================================================================
# CONCEPT 10: STACK-BASED PROBLEMS
# =============================================================================
"""
KEY POINTS:
- Stack: LIFO (Last In, First Out)
- Python list as stack: append() to push, pop() to remove
- Common patterns: matching pairs, nested structures
- Bracket matching: push opening, pop for closing, check match
- Stack is ideal for "most recent" or "nested" problems

CODE EXAMPLES:
    stack = []
    stack.append(item)  # push
    top = stack[-1]     # peek
    item = stack.pop()  # pop
    is_empty = len(stack) == 0
    
    # Bracket matching
    pairs = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != pairs[char]:
                return False
"""

# -----------------------------------------------------------------------------
# PROBLEM 10: Valid Parentheses (LeetCode #20)
# -----------------------------------------------------------------------------
"""
Given a string s containing just the characters '(', ')', '{', '}', 
'[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

Input: s = "([)]"
Output: false

Input: s = "{[]}"
Output: true
"""

def is_valid_parentheses(s: str) -> bool:
    """Stack approach - O(n) time, O(n) space"""
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack.pop() != pairs[char]:
                return False
    
    return len(stack) == 0


def is_valid_parentheses_detailed(s: str) -> bool:
    """More explicit version for interview explanation"""
    stack = []
    mapping = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    
    for char in s:
        # If opening bracket, push to stack
        if char in '({[':
            stack.append(char)
        
        # If closing bracket
        elif char in ')}]':
            # Stack empty = no matching opening bracket
            if not stack:
                return False
            
            # Pop and check if it matches
            top = stack.pop()
            if top != mapping[char]:
                return False
    
    # Valid only if all brackets were matched (stack is empty)
    return len(stack) == 0


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 2: SDET PYTHON INTERVIEW QUESTIONS - TEST CASES")
    print("=" * 60)
    
    # Problem 1: Reverse String
    print("\n1. Reverse String:")
    assert reverse_string_simple("hello") == "olleh"
    s = list("hello")
    reverse_string_inplace(s)
    assert s == list("olleh")
    print("   All test cases passed!")
    
    # Problem 2: Valid Palindrome
    print("\n2. Valid Palindrome:")
    assert is_palindrome_simple("A man, a plan, a canal: Panama") == True
    assert is_palindrome_two_pointer("race a car") == False
    assert is_palindrome_simple(" ") == True
    print("   All test cases passed!")
    
    # Problem 3: Valid Anagram
    print("\n3. Valid Anagram:")
    assert is_anagram_counter("anagram", "nagaram") == True
    assert is_anagram_sort("rat", "car") == False
    assert is_anagram_manual("listen", "silent") == True
    print("   All test cases passed!")
    
    # Problem 4: Contains Duplicate
    print("\n4. Contains Duplicate:")
    assert contains_duplicate([1,2,3,1]) == True
    assert contains_duplicate([1,2,3,4]) == False
    assert contains_duplicate_early_exit([1,1,1,3,3,4,3,2,4,2]) == True
    print("   All test cases passed!")
    
    # Problem 5: Flatten Nested List
    print("\n5. Flatten Nested List:")
    assert flatten_recursive([1, [2, 3], [4, [5, 6]], 7]) == [1,2,3,4,5,6,7]
    assert list(flatten_generator([[1, 2], [3, [4, [5]]]])) == [1,2,3,4,5]
    print("   All test cases passed!")
    
    # Problem 6: Second Largest
    print("\n6. Second Largest Element:")
    assert second_largest_sort([12, 35, 1, 10, 34, 1]) == 34
    assert second_largest_optimal([5, 5, 4, 2]) == 4
    assert second_largest_optimal([10, 10, 10]) == None
    print("   All test cases passed!")
    
    # Problem 7: Top K Frequent
    print("\n7. Top K Frequent Elements:")
    assert set(top_k_frequent([1,1,1,2,2,3], 2)) == {1, 2}
    assert top_k_frequent_bucket([1], 1) == [1]
    print("   All test cases passed!")
    
    # Problem 8: Merge Sorted Arrays
    print("\n8. Merge Sorted Array:")
    nums1 = [1,2,3,0,0,0]
    merge_sorted_inplace(nums1, 3, [2,5,6], 3)
    assert nums1 == [1,2,2,3,5,6]
    assert merge_sorted_new_array([1,3,5], [2,4,6]) == [1,2,3,4,5,6]
    print("   All test cases passed!")
    
    # Problem 9: Missing Number
    print("\n9. Missing Number:")
    assert missing_number_sum([3,0,1]) == 2
    assert missing_number_xor([0,1]) == 2
    assert missing_number_set([9,6,4,2,3,5,7,0,1]) == 8
    print("   All test cases passed!")
    
    # Problem 10: Valid Parentheses
    print("\n10. Valid Parentheses:")
    assert is_valid_parentheses("()") == True
    assert is_valid_parentheses("()[]{}") == True
    assert is_valid_parentheses("(]") == False
    assert is_valid_parentheses("([)]") == False
    assert is_valid_parentheses("{[]}") == True
    print("   All test cases passed!")
    
    print("\n" + "=" * 60)
    print("BLOCK 2 COMPLETE - ALL 10 PROBLEMS SOLVED!")
    print("=" * 60)
