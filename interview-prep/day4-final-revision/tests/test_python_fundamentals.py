from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
reference_root = project_root / "interview-prep" / "day4-final-revision" / "Reference Files"
sys.path.append(str(reference_root))

from block1_python_fundamentals_coding import (
    char_frequency,
    first_non_repeating_char,
    group_anagrams,
    is_anagram,
    is_palindrome,
    reverse_string,
    second_largest,
    valid_parentheses,
)


def test_reverse_string():
    assert reverse_string("hello") == "olleh"


def test_palindrome_ignores_case_and_symbols():
    assert is_palindrome("A man, a plan, a canal: Panama") is True


def test_char_frequency_counts_duplicates():
    assert char_frequency("hello")["l"] == 2


def test_first_non_repeating_char():
    assert first_non_repeating_char("swiss") == "w"


def test_is_anagram():
    assert is_anagram("listen", "silent") is True


def test_second_largest_unique_number():
    assert second_largest([10, 5, 20, 20]) == 10


def test_group_anagrams():
    groups = group_anagrams(["eat", "tea", "tan", "ate"])
    normalized = [sorted(group) for group in groups]
    assert sorted(["ate", "eat", "tea"]) in normalized


def test_valid_parentheses():
    assert valid_parentheses("{[]}") is True
    assert valid_parentheses("{[}]") is False
