from find_solution import (twoSum, minSubArrayLen,
                           longest_substring_without_repeating_characters,
                           find_second_largest, find_nextGreaterElement,
                           longest_repeating_character_replacement, pow_x_n, find_all_subsequences,
                           check_if_power_of_two, divide_two_integers, find_single_num, count_primes, reverse_integer,
                           find_missing_and_repeated_number, assign_cookies, lemonado_change, valid_parenthesis_string,
                           jump_game, jump_game_2, insert_interval, merge_intervals, non_overlapping_intervals)
from find_solution import Solution, TreeNode
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
    assert jump_game_2([2, 3, 0, 1, 4]) == 2
    assert jump_game_2([2, 2, 0, 1, 4]) == 3


def test_insert_interval():
    print(' --- Inside test_insert_interval ---')
    assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]


def test_merge_intervals():
    print(' --- Inside test_merge_intervals ---')
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]


def test_non_overlapping_intervals():
    print(' --- Inside test_non_overlapping_intervals ---')
    assert non_overlapping_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert non_overlapping_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert non_overlapping_intervals([[1, 2], [2, 3]]) == 0


@pytest.fixture
def sample_tree():
    root = TreeNode(1)
    root.right = TreeNode(2)
    root.right.left = TreeNode(3)
    return root


@pytest.fixture
def sample_tree_2():
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)


def test_postorderTraversal(sample_tree):  #left->right->Node
    print(' --- Inside test_postorderTraversal ---')
    assert Solution().postorderTraversal(sample_tree) == [3, 2, 1]


def test_preorderTraversal(sample_tree):  #Node->left->right
    print(' --- Inside test_preorderTraversal ---')
    assert Solution().preorderTraversal(sample_tree) == [1, 2, 3]


def test_inorderTraversal(sample_tree):  #left->node->Right
    print(' --- Inside test_inorderTraversal ---')
    assert Solution().inorderTraversal(sample_tree) == [1, 3, 2]


def test_levelorderTraversal(sample_tree):
    print(' --- Inside test_levelorderTraversal ---')
    assert Solution().levelorderTraversal(sample_tree) == [[1], [2], [3]]


def test_depth_of_binary_tree(sample_tree):
    print(' --- Inside test_depth_of_binary_tree ---')
    assert Solution().depth_of_binary_tree(sample_tree) == 3


def test_balanced_binary_tree(sample_tree, sample_tree_2):
    print(' --- Inside test_balanced_binary_tree ---')
    assert Solution().balanced_binary_tree(sample_tree) == False
    assert Solution().balanced_binary_tree(sample_tree_2) == True


def test_diameter_of_binary_tree(sample_tree_2):
    print(' --- Inside test_diameter_of_binary_tree ---')
    assert Solution().diameter_of_binary_tree(sample_tree_2) == 4


def test_identical_trees_or_not(sample_tree, sample_tree_2):
    print(' --- Inside test_zigzagLevelOrder ---')
    assert Solution().identical_trees_or_not(sample_tree, sample_tree) == True
    assert Solution().identical_trees_or_not(sample_tree, sample_tree_2) == False


def test_zigzagLevelOrder(sample_tree_2):
    print(' --- Inside test_zigzagLevelOrder ---')
    assert Solution().zigzagLevelOrder(sample_tree_2) == []


def test_search_in_bst(sample_tree):
    print(' --- Inside test_search_in_bst ---')
    assert Solution().search_in_bst(sample_tree, 2) == []


def test_max_sum_path():
    print(' --- Inside test_max_sum_path ---')
    assert Solution().max_sum_path([1, 2, 8], [1, 2, 3, 4]) == 11
    assert Solution().max_sum_path([1, 2, 3, 4, 7, 7, 12, 18, 19], [3, 4, 7, 7, 14, 18, 19]) == 75


def test_subsetXORSum():
    print(' --- Inside test_max_sum_path ---')
    assert Solution().subsetXORSum([1, 3]) == 6
    assert Solution().subsetXORSum([5, 1, 6]) == 28
    assert Solution().subsetXORSum([3, 4, 5, 6, 7, 8]) == 480


def test_find_subset():
    print(' --- Inside test_max_sum_path ---')
    assert Solution().find_subset([1, 2]) == [[], [2], [1], [1, 2]]
    assert Solution().find_subset([1, 2, 3]) == [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]


def test_hourglassSum():
    print(' --- Inside test_hourglassSum ---')
    assert Solution().hourglassSum(
        [[1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 2, 4, 4, 0], [0, 0, 0, 2, 0, 0],
         [0, 0, 1, 2, 4, 0]]) == 19
def test_minOperationsToMakeK():
    print(' --- Inside test_minOperationsToMakeK ---')
    assert Solution().minOperationsToMakeK([5,2,5,4,5], 2) == 2
    assert Solution().minOperationsToMakeK([2,1,2], 2) == -1

def test_maximumBeauty():
    print(' --- Inside test_maximumBeauty ---')
    assert Solution().maximumBeauty([[1,2],[3,2],[2,4],[5,6],[3,5]], [1,2,3,4,5,6]) == [2,4,5,5,6,6]

def test_kthSmallest(sample_tree_2):
    print(' --- Inside test_maximumBeauty ---')
    assert Solution().kthSmallest(sample_tree_2, 1) == 5

def test_countPairs():
    print(' --- Inside test_countPairs ---')
    assert Solution().countPairs([3,1,2,2,2,1,3], 2) == 4
    assert Solution().countPairs([1,2,3,4], 1) == 0

def test_countAndSay():
    print(' --- Inside test_countAndSay ---')
    assert Solution().countAndSay(4) == "1211"
    assert Solution().countAndSay(1) == "1"

def test_countFairPairs():
    print(' --- Inside test_countFairPairs ---')
    assert Solution().countFairPairs([0,1,7,4,4,5], 3, 6) == 6
    assert Solution().countFairPairs([1,7,9,2,5], 11, 11) == 1