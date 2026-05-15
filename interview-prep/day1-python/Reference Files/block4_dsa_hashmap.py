"""
================================================================================
BLOCK 4: DSA - HASHMAP & ARRAYS (1.5 hrs)
================================================================================
HashMap (Dictionary) based problems - Most common in coding interviews
Master the O(1) lookup pattern
================================================================================
"""

# =============================================================================
# CONCEPT 1: HASHMAP BASICS
# =============================================================================
"""
KEY POINTS:
- HashMap = Dictionary in Python
- O(1) average time for: lookup, insert, delete
- Use for: counting, caching, mapping relationships
- dict.get(key, default) - safe access with default value
- dict.setdefault(key, default) - get or set if missing
- defaultdict(type) - auto-initializes missing keys
- Counter(iterable) - specialized dict for counting

CODE EXAMPLES:
    # Basic operations
    d = {}
    d['key'] = 'value'     # O(1) insert
    val = d['key']         # O(1) lookup
    del d['key']           # O(1) delete
    'key' in d             # O(1) membership check
    
    # Safe access
    d.get('missing', 'default')  # Returns 'default' if key missing
    
    # Default dict
    from collections import defaultdict
    dd = defaultdict(list)
    dd['key'].append(1)    # Auto-creates empty list if key missing
    
    # Counter
    from collections import Counter
    c = Counter([1, 1, 2, 3, 3, 3])  # {3: 3, 1: 2, 2: 1}
    c.most_common(2)  # [(3, 3), (1, 2)]

HASH COLLISION:
- When two keys hash to same bucket
- Python handles this internally with probing
- Worst case O(n), but extremely rare

HASHABLE OBJECTS:
- Immutable types: int, float, str, tuple (of hashables)
- NOT hashable: list, dict, set (mutable)
- Custom objects: need __hash__ and __eq__
"""


# =============================================================================
# CONCEPT 2: COMPLEMENT PATTERN
# =============================================================================
"""
KEY POINTS:
- "Find pair that sums to target" → HashMap with complement
- complement = target - current
- Store seen values in HashMap for O(1) lookup
- Can return indices or values
- Single pass solution possible

CODE EXAMPLES:
    # Complement pattern
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]  # Found pair!
        seen[num] = i
"""


# -----------------------------------------------------------------------------
# PROBLEM 1: Two Sum (LeetCode #1) - THE MOST CLASSIC PROBLEM
# -----------------------------------------------------------------------------
"""
Given an array of integers nums and an integer target, return indices 
of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, 
and you may not use the same element twice.

You can return the answer in any order.

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9

Input: nums = [3,2,4], target = 6
Output: [1,2]
Explanation: nums[1] + nums[2] = 2 + 4 = 6

Input: nums = [3,3], target = 6
Output: [0,1]

APPROACHES:
1. Brute force: O(n²) - check all pairs
2. HashMap: O(n) - store complement and lookup
"""

def two_sum_brute(nums: list[int], target: int) -> list[int]:
    """Brute force - O(n²) time, O(1) space"""
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def two_sum(nums: list[int], target: int) -> list[int]:
    """HashMap approach - O(n) time, O(n) space"""
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []


def two_sum_all_pairs(nums: list[int], target: int) -> list[list[int]]:
    """Find ALL pairs that sum to target"""
    seen = {}
    result = []
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            for j in seen[complement]:
                result.append([j, i])
        
        if num not in seen:
            seen[num] = []
        seen[num].append(i)
    
    return result


# =============================================================================
# CONCEPT 3: GROUPING PATTERN
# =============================================================================
"""
KEY POINTS:
- Group items by some property → HashMap with lists
- Key = grouping property, Value = list of items
- defaultdict(list) is perfect for this
- Sort-based keys for anagrams
- Use tuple() for hashable keys from lists

CODE EXAMPLES:
    from collections import defaultdict
    
    # Group by first letter
    groups = defaultdict(list)
    for word in words:
        groups[word[0]].append(word)
    
    # Group anagrams (sorted letters as key)
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
"""


# -----------------------------------------------------------------------------
# PROBLEM 2: Group Anagrams (LeetCode #49)
# -----------------------------------------------------------------------------
"""
Given an array of strings strs, group the anagrams together. 
You can return the answer in any order.

An anagram is a word formed by rearranging letters of another word,
using all original letters exactly once.

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]

APPROACH:
- Anagrams have same sorted letters
- Use sorted letters as HashMap key
- Group words with same key
"""

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """Sort-based grouping - O(n * k log k) where k = max string length"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Sorted letters as key (converted to tuple for hashability)
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())


def group_anagrams_count(strs: list[str]) -> list[list[str]]:
    """Character count as key - O(n * k) - faster for long strings"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for s in strs:
        # Count of each character (26 letters) as key
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        groups[tuple(count)].append(s)
    
    return list(groups.values())


# =============================================================================
# CONCEPT 4: EXISTENCE / MEMBERSHIP PATTERN
# =============================================================================
"""
KEY POINTS:
- "Does X exist in collection?" → Set or HashMap
- set() for pure existence checks - O(1)
- dict for existence + associated data
- Converting list to set: O(n) once, then O(1) lookups
- set operations: union |, intersection &, difference -

CODE EXAMPLES:
    # Check duplicates
    seen = set()
    for num in nums:
        if num in seen:
            return True  # Duplicate found!
        seen.add(num)
    
    # Faster way
    has_duplicate = len(nums) != len(set(nums))
    
    # Set operations
    a = {1, 2, 3}
    b = {2, 3, 4}
    a | b  # {1, 2, 3, 4} union
    a & b  # {2, 3} intersection
    a - b  # {1} difference
"""


# -----------------------------------------------------------------------------
# PROBLEM 3: Contains Duplicate (LeetCode #217)
# -----------------------------------------------------------------------------
"""
Given an integer array nums, return true if any value appears 
at least twice in the array, and return false if every element is distinct.

Input: nums = [1,2,3,1]
Output: true

Input: nums = [1,2,3,4]
Output: false

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
"""

def contains_duplicate(nums: list[int]) -> bool:
    """Set length comparison - O(n)"""
    return len(nums) != len(set(nums))


def contains_duplicate_early_exit(nums: list[int]) -> bool:
    """Early exit on finding duplicate - O(n) average"""
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


def contains_duplicate_sorting(nums: list[int]) -> bool:
    """Sorting approach - O(n log n) time, O(1) space"""
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1]:
            return True
    return False


# -----------------------------------------------------------------------------
# PROBLEM 4: Valid Anagram (LeetCode #242)
# -----------------------------------------------------------------------------
"""
Given two strings s and t, return true if t is an anagram of s, 
and false otherwise.

Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false

Follow up: What if the inputs contain Unicode characters?
"""

def is_anagram(s: str, t: str) -> bool:
    """Counter comparison - O(n)"""
    from collections import Counter
    return Counter(s) == Counter(t)


def is_anagram_sort(s: str, t: str) -> bool:
    """Sorting - O(n log n)"""
    return sorted(s) == sorted(t)


def is_anagram_array(s: str, t: str) -> bool:
    """Array counting (only for lowercase a-z) - O(n)"""
    if len(s) != len(t):
        return False
    
    count = [0] * 26
    
    for i in range(len(s)):
        count[ord(s[i]) - ord('a')] += 1
        count[ord(t[i]) - ord('a')] -= 1
    
    return all(c == 0 for c in count)


def is_anagram_unicode(s: str, t: str) -> bool:
    """Works with Unicode - O(n)"""
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
# CONCEPT 5: FREQUENCY COUNTING PATTERN
# =============================================================================
"""
KEY POINTS:
- Count occurrences of items → Counter or dict
- Counter.most_common(k) for top k items
- Find first/unique: count and check
- OrderedDict maintains insertion order (Python 3.7+ dict does too)
- enumerate() gives position for "first occurrence" problems

CODE EXAMPLES:
    from collections import Counter
    
    # Basic counting
    count = Counter(items)
    
    # Manual counting
    count = {}
    for item in items:
        count[item] = count.get(item, 0) + 1
    
    # Find first unique
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
"""


# -----------------------------------------------------------------------------
# PROBLEM 5: First Unique Character in String (LeetCode #387)
# -----------------------------------------------------------------------------
"""
Given a string s, find the first non-repeating character in it 
and return its index. If it does not exist, return -1.

Input: s = "leetcode"
Output: 0
Explanation: 'l' is the first unique character

Input: s = "loveleetcode"
Output: 2
Explanation: 'l' and 'o' repeat, 'v' is first unique at index 2

Input: s = "aabb"
Output: -1
Explanation: No unique character
"""

def first_uniq_char(s: str) -> int:
    """Counter approach - O(n)"""
    from collections import Counter
    
    count = Counter(s)
    
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    
    return -1


def first_uniq_char_dict(s: str) -> int:
    """Manual dict - O(n)"""
    count = {}
    
    # First pass: count occurrences
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    # Second pass: find first with count 1
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    
    return -1


def first_uniq_char_index(s: str) -> int:
    """Using string methods - O(n * 26) for lowercase only"""
    min_index = len(s)
    
    for char in 'abcdefghijklmnopqrstuvwxyz':
        first = s.find(char)
        last = s.rfind(char)
        
        # Unique if first == last and exists
        if first != -1 and first == last:
            min_index = min(min_index, first)
    
    return min_index if min_index < len(s) else -1


# =============================================================================
# BONUS PROBLEMS
# =============================================================================

# -----------------------------------------------------------------------------
# BONUS 1: Two Sum II - Sorted Array (LeetCode #167)
# -----------------------------------------------------------------------------
"""
Given a 1-indexed sorted array, find two numbers that add up to target.
Return indices [index1, index2] where 1 <= index1 < index2.

Input: numbers = [2,7,11,15], target = 9
Output: [1,2] (1-indexed)

Since it's sorted, we can use two pointers instead of HashMap!
"""

def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """Two pointer approach for sorted array - O(n) time, O(1) space"""
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []


# -----------------------------------------------------------------------------
# BONUS 2: Three Sum (LeetCode #15)
# -----------------------------------------------------------------------------
"""
Given array nums, find all unique triplets [a, b, c] where a + b + c = 0.

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Approach: Sort, then for each element, use two pointers for remaining two.
"""

def three_sum(nums: list[int]) -> list[list[int]]:
    """Sort + Two Pointers - O(n²)"""
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        # Two pointers for remaining elements
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result


# -----------------------------------------------------------------------------
# BONUS 3: Longest Consecutive Sequence (LeetCode #128)
# -----------------------------------------------------------------------------
"""
Given unsorted array nums, find length of longest consecutive sequence.
Must run in O(n) time.

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: [1, 2, 3, 4] is longest consecutive sequence.

Approach: Use set for O(1) lookup. Only start counting from sequence start.
"""

def longest_consecutive(nums: list[int]) -> int:
    """Set-based approach - O(n)"""
    if not nums:
        return 0
    
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Only start counting if num is start of sequence
        # (num - 1 not in set means num is start)
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest = max(longest, current_length)
    
    return longest


# =============================================================================
# HASHMAP PATTERN CHEAT SHEET
# =============================================================================
"""
PATTERN RECOGNITION:

1. "Find pair/triplet with sum X" → Complement pattern (target - current)
2. "Group similar items" → HashMap with lists, sorted/counted key
3. "Check if exists/seen before" → Set or HashMap
4. "Count occurrences" → Counter or manual dict
5. "Find first/unique" → Count then iterate in order
6. "Two Sum on sorted" → Two pointers (no HashMap needed)

TIME COMPLEXITY:
- HashMap insert/lookup/delete: O(1) average
- Building HashMap from n items: O(n)
- Sorting: O(n log n)

SPACE COMPLEXITY:
- HashMap with n items: O(n)
- Set with n items: O(n)

WHEN NOT TO USE HASHMAP:
- Sorted array: Two pointers may be O(1) space
- Small fixed range: Array counting may be faster
- Need order: List or OrderedDict
"""


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 4: DSA HASHMAP - TEST CASES")
    print("=" * 60)
    
    # Problem 1: Two Sum
    print("\n1. Two Sum:")
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    print("   All test cases passed!")
    
    # Problem 2: Group Anagrams
    print("\n2. Group Anagrams:")
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Convert to sets for comparison (order doesn't matter)
    result_sets = [set(group) for group in result]
    assert {"eat", "tea", "ate"} in result_sets
    assert {"tan", "nat"} in result_sets
    assert {"bat"} in result_sets
    print("   All test cases passed!")
    
    # Problem 3: Contains Duplicate
    print("\n3. Contains Duplicate:")
    assert contains_duplicate([1, 2, 3, 1]) == True
    assert contains_duplicate([1, 2, 3, 4]) == False
    assert contains_duplicate_early_exit([1, 1, 1, 3, 3, 4]) == True
    print("   All test cases passed!")
    
    # Problem 4: Valid Anagram
    print("\n4. Valid Anagram:")
    assert is_anagram("anagram", "nagaram") == True
    assert is_anagram("rat", "car") == False
    assert is_anagram_array("listen", "silent") == True
    print("   All test cases passed!")
    
    # Problem 5: First Unique Character
    print("\n5. First Unique Character:")
    assert first_uniq_char("leetcode") == 0
    assert first_uniq_char("loveleetcode") == 2
    assert first_uniq_char("aabb") == -1
    print("   All test cases passed!")
    
    # Bonus: Two Sum Sorted
    print("\n6. Bonus - Two Sum Sorted:")
    assert two_sum_sorted([2, 7, 11, 15], 9) == [1, 2]
    print("   All test cases passed!")
    
    # Bonus: Three Sum
    print("\n7. Bonus - Three Sum:")
    result = three_sum([-1, 0, 1, 2, -1, -4])
    assert [-1, -1, 2] in result
    assert [-1, 0, 1] in result
    print("   All test cases passed!")
    
    # Bonus: Longest Consecutive
    print("\n8. Bonus - Longest Consecutive Sequence:")
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    print("   All test cases passed!")
    
    print("\n" + "=" * 60)
    print("BLOCK 4 COMPLETE - ALL HASHMAP PROBLEMS SOLVED!")
    print("=" * 60)
