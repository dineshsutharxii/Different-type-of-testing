from find_solution import (twoSum, minSubArrayLen,
                           longest_substring_without_repeating_characters,
                           find_second_largest, find_nextGreaterElement)
import pytest

def test_twosum():
    print(" --- Inside test_twosum ---")
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]
    assert twoSum([3, 2, 4], 6) == [1, 2]
    assert twoSum([3, 3], 6) == [0, 1]


def test_minSubArrayLen():
    print(" --- Inside test_minSubArrayLen ---")
    assert minSubArrayLen([2, 3, 1, 2, 4, 3], 7) == 2
    assert minSubArrayLen([1, 4, 4], 4) == 1
    assert minSubArrayLen([1, 1, 1, 1, 1, 1, 1, 1], 11) == 0

def test_longest_substring_without_repeating_characters():
    print(" --- Inside test_longest_substring_without_repeating_characters ---")
    assert longest_substring_without_repeating_characters("abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") == "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert longest_substring_without_repeating_characters("abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZZ0123456789") == "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def test_find_second_largest():
    print(" --- Inside test_find_second_largest ---")
    assert find_second_largest([2, 3, 1, 2, 4, 3]) == 3

def test_nextGreaterElement():
    print(" --- Inside test_nextGreaterElement ---")
    assert find_nextGreaterElement([4,1,2], [1,3,4,2]) == [-1,3,-1]


