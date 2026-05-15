"""
================================================================================
BLOCK 5: DSA - TWO POINTERS (1 hr)
================================================================================
Two pointer technique - Essential pattern for array/string problems
Often turns O(n²) brute force into O(n) optimal solution
================================================================================
"""

# =============================================================================
# CONCEPT 1: TWO POINTERS - OPPOSITE ENDS
# =============================================================================
"""
KEY POINTS:
- Start with left at beginning, right at end
- Move pointers inward based on condition
- Perfect for: palindromes, sorted array problems, container problems
- Typically O(n) time, O(1) space
- Works when array is sorted OR symmetric check needed

CODE EXAMPLES:
    # Basic pattern
    left, right = 0, len(arr) - 1
    
    while left < right:
        # Process arr[left] and arr[right]
        
        if condition_to_move_left:
            left += 1
        else:
            right -= 1
    
    # Palindrome check
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

WHEN TO USE:
- "Find pair in sorted array"
- "Check if string is palindrome"
- "Container/area problems"
- "Reverse in place"
"""


# -----------------------------------------------------------------------------
# PROBLEM 1: Valid Palindrome (LeetCode #125)
# -----------------------------------------------------------------------------
"""
A phrase is a palindrome if, after converting all uppercase letters 
to lowercase and removing all non-alphanumeric characters, it reads 
the same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Input: s = " "
Output: true
Explanation: Empty string after cleaning is palindrome.

APPROACHES:
1. Clean string, then compare with reverse - O(n) time, O(n) space
2. Two pointers with in-place skip - O(n) time, O(1) space
"""

def is_palindrome_clean(s: str) -> bool:
    """Clean first, then reverse - O(n) space"""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointer(s: str) -> bool:
    """Two pointers - O(1) space"""
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


# -----------------------------------------------------------------------------
# PROBLEM 2: Valid Palindrome II (LeetCode #680)
# -----------------------------------------------------------------------------
"""
Given a string s, return true if s can be palindrome after deleting 
at most one character from it.

Input: s = "aba"
Output: true

Input: s = "abca"
Output: true
Explanation: Delete 'c' or 'b' to get "aba" or "aca".

Input: s = "abc"
Output: false

This is a follow-up question often asked after basic palindrome!
"""

def valid_palindrome_with_deletion(s: str) -> bool:
    """Two pointers with one deletion allowed"""
    
    def is_palindrome_range(left: int, right: int) -> bool:
        """Check if s[left:right+1] is palindrome"""
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try skipping left OR skipping right
            return (is_palindrome_range(left + 1, right) or 
                    is_palindrome_range(left, right - 1))
        left += 1
        right -= 1
    
    return True


# =============================================================================
# CONCEPT 2: TWO POINTERS - SLIDING/SHRINKING WINDOW
# =============================================================================
"""
KEY POINTS:
- Two pointers define a window/range
- Move pointers to maximize/minimize window based on condition
- Often combined with greedy approach
- Track best result seen so far

CODE EXAMPLES:
    # Container/area pattern
    left, right = 0, len(arr) - 1
    max_area = 0
    
    while left < right:
        # Calculate current window's value
        current = calculate(arr[left], arr[right], right - left)
        max_area = max(max_area, current)
        
        # Shrink from the smaller side (greedy)
        if arr[left] < arr[right]:
            left += 1
        else:
            right -= 1
"""


# -----------------------------------------------------------------------------
# PROBLEM 3: Container With Most Water (LeetCode #11)
# -----------------------------------------------------------------------------
"""
Given n non-negative integers height where each represents a vertical 
line at position i, find two lines that together with the x-axis forms 
a container that holds the most water.

Return the maximum amount of water a container can store.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: Lines at index 1 (height 8) and index 8 (height 7).
             Width = 8-1 = 7, Height = min(8,7) = 7
             Area = 7 * 7 = 49

Input: height = [1,1]
Output: 1

INTUITION:
- Area = width × min(height[left], height[right])
- Start with maximum width (left=0, right=n-1)
- Moving the shorter line might find a taller one → more area
- Moving the taller line can only decrease area (less width, same or less height)
"""

def max_area_brute(height: list[int]) -> int:
    """Brute force - O(n²)"""
    n = len(height)
    max_water = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            width = j - i
            h = min(height[i], height[j])
            max_water = max(max_water, width * h)
    
    return max_water


def max_area(height: list[int]) -> int:
    """Two pointers greedy - O(n) time, O(1) space"""
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Calculate current area
        width = right - left
        h = min(height[left], height[right])
        current_area = width * h
        max_water = max(max_water, current_area)
        
        # Move the shorter line (greedy choice)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water


# =============================================================================
# CONCEPT 3: TWO POINTERS - SLOW/FAST (SAME DIRECTION)
# =============================================================================
"""
KEY POINTS:
- Both pointers start from same end, move same direction
- Slow pointer tracks "valid" position
- Fast pointer scans ahead
- Used for: removing elements, deduplication, partitioning
- In-place modification pattern

CODE EXAMPLES:
    # Remove duplicates from sorted array
    slow = 0  # Position to write next unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Length of unique portion

WHEN TO USE:
- "Remove element in place"
- "Remove duplicates from sorted array"
- "Move zeros to end"
- "Partition array"
"""


# -----------------------------------------------------------------------------
# PROBLEM 4: Remove Duplicates from Sorted Array (LeetCode #26)
# -----------------------------------------------------------------------------
"""
Given an integer array nums sorted in non-decreasing order, remove the 
duplicates in-place such that each unique element appears only once. 
The relative order of the elements should be kept the same.

Return k = the number of unique elements.

The first k elements of nums should hold the unique elements in their 
original order. Elements beyond k don't matter.

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]

KEY INSIGHT:
- Since sorted, duplicates are adjacent
- Slow pointer: last unique element position
- Fast pointer: scans for next unique element
"""

def remove_duplicates(nums: list[int]) -> int:
    """Slow/fast pointer - O(n) time, O(1) space"""
    if not nums:
        return 0
    
    slow = 0  # Position of last unique element
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1  # Number of unique elements


def remove_duplicates_k(nums: list[int], k: int = 2) -> int:
    """Allow at most k duplicates (LeetCode #80) - O(n)"""
    if len(nums) <= k:
        return len(nums)
    
    slow = k  # First k elements always valid
    
    for fast in range(k, len(nums)):
        if nums[fast] != nums[slow - k]:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow


# -----------------------------------------------------------------------------
# PROBLEM 5: Move Zeroes (LeetCode #283)
# -----------------------------------------------------------------------------
"""
Given an integer array nums, move all 0's to the end while maintaining 
the relative order of the non-zero elements.

You must do this in-place without making a copy of the array.

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]
"""

def move_zeroes(nums: list[int]) -> None:
    """Slow/fast pointer - O(n) time, O(1) space"""
    slow = 0  # Position to place next non-zero
    
    # Move all non-zeros to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    
    # Fill remaining with zeros
    while slow < len(nums):
        nums[slow] = 0
        slow += 1


def move_zeroes_swap(nums: list[int]) -> None:
    """Swap approach - fewer writes"""
    slow = 0
    
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# -----------------------------------------------------------------------------
# PROBLEM 6: Remove Element (LeetCode #27)
# -----------------------------------------------------------------------------
"""
Given an array nums and a value val, remove all instances of val in-place 
and return the new length.

Order can be changed. Elements beyond returned length don't matter.

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
"""

def remove_element(nums: list[int], val: int) -> int:
    """Slow/fast pointer - O(n)"""
    slow = 0
    
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    
    return slow


def remove_element_swap(nums: list[int], val: int) -> int:
    """Swap with end - fewer operations when val is rare"""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        if nums[left] == val:
            nums[left] = nums[right]
            right -= 1
        else:
            left += 1
    
    return left


# =============================================================================
# BONUS: THREE POINTERS / ADVANCED
# =============================================================================

# -----------------------------------------------------------------------------
# BONUS 1: Sort Colors (LeetCode #75) - Dutch National Flag
# -----------------------------------------------------------------------------
"""
Given array with n objects colored red, white, or blue (0, 1, 2),
sort them in-place so same colors are adjacent: red, white, blue.

Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

This is the famous Dutch National Flag problem by Dijkstra.
Uses THREE pointers: low, mid, high
"""

def sort_colors(nums: list[int]) -> None:
    """Dutch National Flag - O(n) time, O(1) space"""
    low = 0       # Boundary for 0s
    mid = 0       # Current element
    high = len(nums) - 1  # Boundary for 2s
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid - need to check swapped element


# -----------------------------------------------------------------------------
# BONUS 2: Trapping Rain Water (LeetCode #42)
# -----------------------------------------------------------------------------
"""
Given n non-negative integers representing elevation map where width 
of each bar is 1, compute how much water it can trap after raining.

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

This is a HARD problem but demonstrates advanced two pointers!
"""

def trap_water(height: list[int]) -> int:
    """Two pointers - O(n) time, O(1) space"""
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    
    return water


# -----------------------------------------------------------------------------
# BONUS 3: Reverse Words in a String (LeetCode #151)
# -----------------------------------------------------------------------------
"""
Given a string s, reverse the order of words.
A word is a sequence of non-space characters.

Input: s = "the sky is blue"
Output: "blue is sky the"

Input: s = "  hello world  "
Output: "world hello"
"""

def reverse_words(s: str) -> str:
    """Pythonic solution"""
    return ' '.join(s.split()[::-1])


def reverse_words_manual(s: str) -> str:
    """Manual approach showing two-pointer string handling"""
    # Split into words (handling multiple spaces)
    words = []
    i = 0
    n = len(s)
    
    while i < n:
        # Skip spaces
        while i < n and s[i] == ' ':
            i += 1
        
        if i >= n:
            break
        
        # Find word end
        j = i
        while j < n and s[j] != ' ':
            j += 1
        
        words.append(s[i:j])
        i = j
    
    # Reverse and join
    words.reverse()
    return ' '.join(words)


# =============================================================================
# TWO POINTER PATTERN CHEAT SHEET
# =============================================================================
"""
PATTERN RECOGNITION:

1. OPPOSITE ENDS (left=0, right=n-1):
   - Palindrome check
   - Two sum in sorted array
   - Container with most water
   - Reverse array/string

2. SLOW/FAST (same direction):
   - Remove duplicates (sorted)
   - Remove element
   - Move zeros
   - Partition array

3. EXPAND FROM CENTER:
   - Longest palindromic substring
   - Find all palindromes

4. SLIDING WINDOW (related):
   - Subarray sum
   - Longest substring with k chars

KEY INSIGHTS:
- Sorted array → likely two pointers
- In-place modification → slow/fast
- Maximize/minimize with constraints → shrinking window
- Symmetric property → opposite ends

COMPLEXITY:
- Usually O(n) time
- Usually O(1) space (in-place)
"""


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCK 5: DSA TWO POINTERS - TEST CASES")
    print("=" * 60)
    
    # Problem 1: Valid Palindrome
    print("\n1. Valid Palindrome:")
    assert is_palindrome_two_pointer("A man, a plan, a canal: Panama") == True
    assert is_palindrome_two_pointer("race a car") == False
    assert is_palindrome_two_pointer(" ") == True
    print("   All test cases passed!")
    
    # Problem 2: Valid Palindrome II
    print("\n2. Valid Palindrome II (with deletion):")
    assert valid_palindrome_with_deletion("aba") == True
    assert valid_palindrome_with_deletion("abca") == True
    assert valid_palindrome_with_deletion("abc") == False
    print("   All test cases passed!")
    
    # Problem 3: Container With Most Water
    print("\n3. Container With Most Water:")
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area([1, 1]) == 1
    assert max_area([4, 3, 2, 1, 4]) == 16
    print("   All test cases passed!")
    
    # Problem 4: Remove Duplicates
    print("\n4. Remove Duplicates from Sorted Array:")
    nums = [1, 1, 2]
    k = remove_duplicates(nums)
    assert k == 2 and nums[:k] == [1, 2]
    
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k = remove_duplicates(nums)
    assert k == 5 and nums[:k] == [0, 1, 2, 3, 4]
    print("   All test cases passed!")
    
    # Problem 5: Move Zeroes
    print("\n5. Move Zeroes:")
    nums = [0, 1, 0, 3, 12]
    move_zeroes(nums)
    assert nums == [1, 3, 12, 0, 0]
    print("   All test cases passed!")
    
    # Problem 6: Remove Element
    print("\n6. Remove Element:")
    nums = [3, 2, 2, 3]
    k = remove_element(nums, 3)
    assert k == 2 and sorted(nums[:k]) == [2, 2]
    print("   All test cases passed!")
    
    # Bonus: Sort Colors
    print("\n7. Bonus - Sort Colors (Dutch Flag):")
    nums = [2, 0, 2, 1, 1, 0]
    sort_colors(nums)
    assert nums == [0, 0, 1, 1, 2, 2]
    print("   All test cases passed!")
    
    # Bonus: Trapping Rain Water
    print("\n8. Bonus - Trapping Rain Water:")
    assert trap_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap_water([4, 2, 0, 3, 2, 5]) == 9
    print("   All test cases passed!")
    
    # Bonus: Reverse Words
    print("\n9. Bonus - Reverse Words:")
    assert reverse_words("the sky is blue") == "blue is sky the"
    assert reverse_words("  hello world  ") == "world hello"
    print("   All test cases passed!")
    
    print("\n" + "=" * 60)
    print("BLOCK 5 COMPLETE - ALL TWO POINTER PROBLEMS SOLVED!")
    print("=" * 60)
