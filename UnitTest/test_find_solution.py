from find_solution import (twoSum, minSubArrayLen,
                           longest_substring_without_repeating_characters,
                           find_second_largest, find_nextGreaterElement,
                           longest_repeating_character_replacement, pow_x_n, find_all_subsequences,
                           check_if_power_of_two, divide_two_integers, find_single_num, count_primes, reverse_integer,
                           find_missing_and_repeated_number, assign_cookies, lemonado_change, valid_parenthesis_string,
                           jump_game, jump_game_2)
import pytest


def test_twosum():
    print(" --- Inside test_twosum ---")
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]
    assert twoSum([3, 2, 4], 6) == [1, 2]
    assert twoSum([3, 3], 6) == [0, 1]
    assert twoSum([3, 3, 7, 6], 13) == [2, 3]


def test_minSubArrayLen():
    print(" --- Inside test_minSubArrayLen ---")
    assert minSubArrayLen([2, 3, 1, 2, 4, 3], 7) == 2
    assert minSubArrayLen([1, 4, 4], 4) == 1
    assert minSubArrayLen([1, 1, 1, 1, 1, 1, 1, 1], 11) == 0


def test_longest_substring_without_repeating_characters():
    print(" --- Inside test_longest_substring_without_repeating_characters ---")
    assert longest_substring_without_repeating_characters(
        "abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") == "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    assert longest_substring_without_repeating_characters(
        "abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZZ0123456789") == "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_find_second_largest():
    print(" --- Inside test_find_second_largest ---")
    assert find_second_largest([2, 3, 1, 2, 4, 3]) == 3
    assert find_second_largest([2, 3, 1, 2, 4, 5]) == 4


def test_nextGreaterElement():
    print(" --- Inside test_nextGreaterElement ---")
    assert find_nextGreaterElement([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
    assert find_nextGreaterElement([2, 4], [1, 2, 3, 4]) == [3, -1]


def test_longest_repeating_character_replacement():
    print(" --- Inside test_longest_repeating_character_replacement ---")
    assert longest_repeating_character_replacement("ABAB", 2) == 4


def test_pow_x_n():
    print(" --- Inside test_pow_x_n ---")
    assert pow_x_n(2.00, 10) == 1024.0000
    assert pow_x_n(2.10000, 3) == 9.26100
    assert pow_x_n(2.00000, -2) == 0.25000


def test_find_all_subsequences():
    print(" --- Inside test_find_all_subsequences ---")
    assert find_all_subsequences([1, 2, 3]) == [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    assert find_all_subsequences([0]) == [[], [0]]


def test_check_if_power_of_two():
    print(" --- Inside test_check_if_power_of_two ---")
    assert check_if_power_of_two(16) == True
    assert check_if_power_of_two(1) == True
    assert check_if_power_of_two(6) == False


def test_divide_two_integers():
    print(" --- Inside test_divide_two_integers ---")
    assert divide_two_integers(10, 3) == 3
    assert divide_two_integers(7, -3) == -2


def test_find_single_num():
    print(" --- Inside test_find_single_num ---")
    assert find_single_num([4, 1, 2, 1, 2]) == 4
    assert find_single_num([2, 2, 1]) == 1
    assert find_single_num([1]) == 1


def test_count_primes():
    print(" --- Inside test_count_primes ---")
    assert count_primes(6) == 3
    assert count_primes(30) == 10
    assert count_primes(0) == 0


def test_reverse_integer():
    print(" --- Inside test_count_primes ---")
    assert reverse_integer(123) == 321
    assert reverse_integer(-123) == -321
    assert reverse_integer(1534236469) == 0
    assert reverse_integer(-1534236469) == 0


def test_find_missing_and_repeated_number():
    print(" --- Inside test_find_missing_and_repeated_number ---")
    assert find_missing_and_repeated_number([[1, 3], [2, 2]]) == [2, 4]
    assert find_missing_and_repeated_number([[9, 1, 7], [8, 9, 2], [3, 4, 6]]) == [9, 5]


def test_assign_cookies():
    print(" --- Inside test_assign_cookies ---")
    assert assign_cookies([1, 2, 3], [1, 1]) == 1
    assert assign_cookies([1, 2], [1, 2, 3]) == 2


def test_lemonado_change():
    print(" --- Inside test_lemonado_change ---")
    assert lemonado_change([5, 5, 5, 10, 20]) == True
    assert lemonado_change([5, 5, 10, 10, 20]) == False


def test_valid_parenthesis_string():
    print(" --- Inside test_valid_parenthesis_string ---")
    assert valid_parenthesis_string("()")
    assert valid_parenthesis_string("(*)")
    assert valid_parenthesis_string("(*))")


def test_jump_game():
    print(' --- Inside test_jump_game ---')
    assert jump_game([2, 3, 1, 1, 4]) == True
    assert jump_game([3, 2, 1, 0, 4]) == False


def test_jump_game_2():
    print(' --- Inside test_jump_game_2 ---')
    assert jump_game_2([2, 3, 1, 1, 4]) == 2
    assert jump_game_2([2,3,0,1,4]) == 2
    assert jump_game_2([2,2,0,1,4]) == 3
