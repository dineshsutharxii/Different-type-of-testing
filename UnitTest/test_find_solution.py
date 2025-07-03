from collections import Counter

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
    assert Solution().minOperationsToMakeK([5, 2, 5, 4, 5], 2) == 2
    assert Solution().minOperationsToMakeK([2, 1, 2], 2) == -1


def test_maximumBeauty():
    print(' --- Inside test_maximumBeauty ---')
    assert Solution().maximumBeauty([[1, 2], [3, 2], [2, 4], [5, 6], [3, 5]], [1, 2, 3, 4, 5, 6]) == [2, 4, 5, 5, 6, 6]


def test_kthSmallest(sample_tree_2):
    print(' --- Inside test_maximumBeauty ---')
    assert Solution().kthSmallest(sample_tree_2, 1) == 5


def test_countPairs():
    print(' --- Inside test_countPairs ---')
    assert Solution().countPairs([3, 1, 2, 2, 2, 1, 3], 2) == 4
    assert Solution().countPairs([1, 2, 3, 4], 1) == 0


def test_countAndSay():
    print(' --- Inside test_countAndSay ---')
    assert Solution().countAndSay(4) == "1211"
    assert Solution().countAndSay(1) == "1"


def test_countFairPairs():
    print(' --- Inside test_countFairPairs ---')
    assert Solution().countFairPairs([0, 1, 7, 4, 4, 5], 3, 6) == 6
    assert Solution().countFairPairs([1, 7, 9, 2, 5], 11, 11) == 1


def test_numRabbits():
    print(' --- Inside test_numRabbits ---')
    assert Solution().numRabbits([1, 1, 2]) == 5
    assert Solution().numRabbits([10, 10, 10]) == 11


def test_numberOfArrays():
    print(' --- Inside test_numberOfArrays ---')
    assert Solution().numberOfArrays([1, -3, 4], 1, 6) == 2
    assert Solution().numberOfArrays([3, -4, 5, 1, -2], -4, 5) == 4


def test_maxMatrixSum():
    print(' --- Inside test_maxMatrixSum ---')
    assert Solution().maxMatrixSum([[1, -1], [-1, 1]]) == 4
    assert Solution().maxMatrixSum([[1, 2, 3], [-1, -2, -3], [1, 2, 3]]) == 16


def test_countLargestGroup():
    print(' --- Inside test_countLargestGroup ---')
    assert Solution().countLargestGroup(13) == 4
    assert Solution().countLargestGroup(2) == 2


def test_countCompleteSubarrays():
    print(' --- Inside test_countCompleteSubarrays ---')
    assert Solution().countCompleteSubarrays([1, 3, 1, 2, 2]) == 4
    assert Solution().countCompleteSubarrays([5, 5, 5, 5]) == 10


def test_countInterestingSubarrays():
    print(' --- Inside test_countInterestingSubarrays ---')
    assert Solution().countInterestingSubarrays([3, 2, 4], 2, 1) == 3


def test_countSubarrays():
    print(' --- Inside test_countSubarrays ---')
    assert Solution().countSubarrays([1, 3, 5, 2, 7, 5], 1, 5) == 2


def test_countSubarraysWithCondition():
    print(' --- Inside test_countSubarrays ---')
    assert Solution().countSubarraysWithCondition([1, 2, 1, 4, 1]) == 1
    assert Solution().countSubarraysWithCondition([1, 1, 1]) == 0


def test_countSubarraysWithScoreLessThanK():
    print(' --- Inside test_countSubarraysWithScoreLessThanK ---')
    assert Solution().countSubarraysWithScoreLessThanK([2, 1, 4, 3, 5], 10)
    assert Solution().countSubarraysWithScoreLessThanK([1, 1, 1], 5)


def test_count_Subarrays_max_Element_Appears_at_Least_K_Times():
    print(' --- Inside test_count_Subarrays_max_Element_Appears_at_Least_K_Times ---')
    assert Solution().count_Subarrays_max_Element_Appears_at_Least_K_Times([1, 3, 2, 3, 3], 2) == 6
    assert Solution().count_Subarrays_max_Element_Appears_at_Least_K_Times([1, 4, 2, 1], 3) == 0


def test_findCountOfNumbersWithEvenString():
    print(' --- Inside test_count_Subarrays_max_Element_Appears_at_Least_K_Times ---')
    assert Solution().findCountOfNumbersWithEvenString([12, 345, 2, 6, 7896]) == 2
    assert Solution().findCountOfNumbersWithEvenString([555, 901, 482, 1771]) == 1


def test_addSpaces():
    print(' --- Inside test_addSpaces ---')
    assert Solution().addSpaces("LeetcodeHelpsMeLearn", [8, 13, 15]) == "Leetcode Helps Me Learn"
    assert Solution().addSpaces("icodeinpython", [1, 5, 7, 9]) == "i code in py thon"


def test_pushDominoes():
    print(' --- Inside test_pushDominoes ---')
    assert Solution().pushDominoes("RR.L") == "RR.L"
    assert Solution().pushDominoes(".L.R...LR..L..") == "LL.RR.LLRRLL.."


def test_minDominoRotations():
    print(' --- Inside test_minDominoRotations ---')
    assert Solution().minDominoRotations([3, 5, 1, 2, 3], [3, 6, 3, 3, 4]) == -1
    assert Solution().minDominoRotations([2, 1, 2, 4, 2, 2], [5, 2, 6, 2, 3, 2]) == 2


def test_numTilings():
    print(' --- Inside test_numTilings ---')
    assert Solution().numTilings(3) == 5
    assert Solution().numTilings(1) == 1
    assert Solution().numTilings(9) == 569


def test_buildArray():
    print(' --- Inside test_buildArray ---')
    assert Solution().buildArray([0, 2, 1, 5, 3, 4]) == [0, 1, 2, 4, 5, 3]
    assert Solution().buildArray([5, 0, 1, 2, 3, 4]) == [4, 5, 0, 1, 2, 3]


def test_minTimeToReach():
    print(' --- Inside test_minTimeToReach ---')
    assert Solution().minTimeToReach([[0, 4], [4, 4]]) == 7
    assert Solution().minTimeToReach([[0, 0, 0, 0], [0, 0, 0, 0]]) == 6


def test_maximumBeauty():
    print(' --- Inside test_maximumBeauty ---')
    assert Solution().maximumBeauty([4, 6, 1, 2], 2) == 3


def test_minSum():
    print(' --- Inside test_minSum ---')
    assert Solution().minSum([3, 2, 0, 1, 0], [6, 5, 0]) == 12
    assert Solution().minSum([2,0,2,0], [1,4]) == -1


def test_threeConsecutiveOdds():
    print(' --- Inside test_threeConsecutiveOdds ---')
    assert Solution().threeConsecutiveOdds([2, 6, 4, 1]) == False
    assert Solution().threeConsecutiveOdds([1, 2, 34, 3, 4, 5, 7, 23, 12]) == True


def test_findScore():
    print(' --- Inside test_findScore ---')
    assert Solution().findScore([2, 1, 3, 4, 5, 2]) == 7
    assert Solution().findScore([2, 3, 5, 1, 3, 2]) == 5


def test_lengthAfterTransformations():
    print(' --- Inside test_lengthAfterTransformations ---')
    assert Solution().lengthAfterTransformations("abcyy", 2) == 7
    assert Solution().lengthAfterTransformations("azbk", 1) == 5


def test_getFinalState():
    print(' --- Inside test_getFinalState ---')
    assert Solution().getFinalState([2, 1, 3, 5, 6], 5, 2) == [8, 4, 6, 5, 6]
    assert Solution().getFinalState([1, 2], 3, 4) == [16, 8]


def test_getLongestSubsequence():
    print(' --- Inside test_getLongestSubsequence ---')
    assert Solution().getLongestSubsequence(["c"], [0]) == ["c"]
    assert Solution().getLongestSubsequence(["e", "a", "b"], [0, 0, 1]) == ["e", "b"]
    assert Solution().getLongestSubsequence(["a", "b", "c", "d"], [1, 0, 1, 1]) == ["a", "b", "c"]


def test_finalPrices():
    print(' --- Inside test_finalPrices ---')
    assert Solution().finalPrices([8, 4, 6, 2, 3]) == [4, 2, 4, 2, 3]
    assert Solution().finalPrices([10, 1, 1, 6]) == [9, 0, 1, 6]
    assert Solution().finalPrices([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_sortColors():
    print(' --- Inside test_finalPrices ---')
    assert Solution().sortColors([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]
    assert Solution().sortColors([2, 0, 1]) == [0, 1, 2]


def test_maxChunksToSorted():
    print(' --- Inside test_maxChunksToSorted ---')
    assert Solution().maxChunksToSorted([4, 3, 2, 1, 0]) == 1
    assert Solution().maxChunksToSorted([1, 0, 2, 3, 4]) == 4


def test_triangleType():
    print(' --- Inside test_triangleType ---')
    assert Solution().triangleType([3, 3, 3]) == "equilateral"
    assert Solution().triangleType([3, 4, 5]) == "scalene"
    assert Solution().triangleType([8, 4, 2]) == "none"


def test_isZeroArray():
    print(' --- Inside test_isZeroArray ---')
    assert Solution().isZeroArray([1, 0, 1], [[0, 2]])
    assert not Solution().isZeroArray([4, 3, 2, 1], [[1, 3], [0, 2]])


def test_setZeroes():
    print(' --- Inside test_setZeroes ---')
    assert Solution().setZeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]


def test_maxRemoval():
    print(' --- Inside test_setZeroes ---')
    assert Solution().maxRemoval([2, 0, 2], [[0, 2], [0, 2], [1, 1]]) == 1
    assert Solution().maxRemoval([1, 2, 3, 4], [[0, 3]]) == -1


def test_findTargetSumWays():
    print(' --- Inside test_findTargetSumWays ---')
    assert Solution().findTargetSumWays([1, 1, 1, 1, 1], 3) == 5
    assert Solution().findTargetSumWays([1], 1) == 1


def test_findWordsContaining():
    print(' --- Inside test_findWordsContaining ---')
    assert Solution().findTargetSumWays(["leet", "code"], "e") == [0, 1]
    assert Solution().findTargetSumWays(["abc", "bcd", "aaaa", "cbc"], "a") == [0, 2]


def test_longestPalindrome():
    print(' --- Inside test_longestPalindrome ---')
    assert Solution().longestPalindrome(["lc", "cl", "gg"]) == 6
    assert Solution().longestPalindrome(["cc", "ll", "xx"]) == 2


def test_maxCount():
    print(' --- Inside test_maxCount ---')
    assert Solution().maxCount("011101") == 5
    assert Solution().maxCount("1111") == 3
    assert Solution().maxCount("10011") == 4


def test_mincostTickets():
    print(' --- Inside test_mincostTickets ---')
    assert Solution().mincostTickets([1, 4, 6, 7, 8, 20], [2, 7, 15]) == 11


def test_waysToSplitArray():
    print(' --- Inside test_waysToSplitArray ---')
    assert Solution().waysToSplitArray([10, 4, -8, 7]) == 2


def test_snakesAndLadders():
    print(' --- Inside test_snakesAndLadders ---')
    assert Solution().snakesAndLadders(
        [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, 35, -1, -1, 13, -1],
         [-1, -1, -1, -1, -1, -1], [-1, 15, -1, -1, -1, -1]]) == 4
    assert Solution().snakesAndLadders([[-1, -1], [-1, 3]]) == 1


def test_distributeCandies():
    print(' --- Inside test_distributeCandies ---')
    assert Solution().distributeCandies(5, 2) == 3
    assert Solution().distributeCandies(3, 3) == 10


def test_findDifferentBinaryString():
    print(' --- Inside test_findDifferentBinaryString ---')
    assert Solution().findDifferentBinaryString(["01", "10"]) == "00"


def test_answerString():
    print(' --- Inside test_answerString ---')
    assert Solution().answerString("dbca", 2) == "dbc"


def test_smallestEquivalentString():
    print(' --- Inside test_answerString ---')
    assert Solution().smallestEquivalentString("parker", "morris", "parser") == "makkek"


def test_clearStars():
    print(' --- Inside test_clearStars ---')
    assert Solution().clearStars("abc") == "abc"
    assert Solution().clearStars("aaba*") == "aab"


def test_lexicalOrder():
    print(' --- Inside test_lexicalOrder ---')
    assert Solution().lexicalOrder(13) == [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]
    assert Solution().lexicalOrder(2) == [1, 2]


def test_canConstruct():
    print(' --- Inside test_canConstruct ---')
    assert Solution().canConstruct("leetcode", 3) == False
    assert Solution().canConstruct("true", 4) == True


def test_minimumLength():
    print(' --- Inside test_minimumLength ---')
    assert Solution().minimumLength("abaacbcbb") == 5
    assert Solution().minimumLength("aa") == 2


def test_maxAdjacentDistance():
    print(' --- Inside test_minimumLength ---')
    assert Solution().maxAdjacentDistance([-2, 1, -5]) == 6
    assert Solution().maxAdjacentDistance([1, 2, 4]) == 3
    assert Solution().maxAdjacentDistance([-5, -10, -5]) == 5


def test_minimizeMax():
    print(' --- Inside test_minimizeMax ---')
    assert Solution().minimizeMax([10, 1, 2, 7, 1, 3], 2) == 1


def test_minMaxDifference():
    print(' --- Inside test_minimizeMax ---')
    assert Solution().minMaxDifference(11891) == 99009
    assert Solution().minMaxDifference(90) == 99


def test_maxDiff():
    print(' --- Inside test_minimizeMax ---')
    assert Solution().maxDiff(555) == 888
    assert Solution().maxDiff(5689) == 8000


def test_maximumDifference():
    print(' --- Inside test_maximumDifference ---')
    assert Solution().maximumDifference([1, 5, 2, 10]) == 9


def test_numOfSubarrays():
    print(' --- Inside test_numOfSubarrays ---')
    assert Solution().numOfSubarrays([1, 3, 5]) == 4
    assert Solution().numOfSubarrays([2, 4, 6]) == 0
    assert Solution().numOfSubarrays([1, 2, 3, 4, 5, 6, 7]) == 16


def test_numOfSubarraysOptimize():
    print(' --- Inside test_numOfSubarraysOptimize ---')
    assert Solution().numOfSubarrays([1, 3, 5]) == 4
    assert Solution().numOfSubarrays([2, 4, 6]) == 0
    assert Solution().numOfSubarrays([1, 2, 3, 4, 5, 6, 7]) == 16


def test_partitionArray():
    print(' --- Inside test_partitionArray ---')
    assert Solution().partitionArray([3, 6, 1, 2, 5], 2) == 2


def test_minimumDeletions():
    print(' --- Inside test_minimumDeletions ---')
    assert Solution().minimumDeletions("aabcaba", 0) == 3
    assert Solution().minimumDeletions("dabdcbdcdcd", 2) == 2


def test_divideString():
    print(' --- Inside test_divideString ---')
    assert Solution().divideString("abcdefghi", 3, "x") == ["abc","def","ghi"]
    assert Solution().divideString("abcdefghij", 3, "x") == ["abc","def","ghi","jxx"]

def test_findKDistantIndices():
    print(' --- Inside test_findKDistantIndices ---')
    assert Solution().findKDistantIndices([2,2,2,2,2], 2, 2) == [0,1,2,3,4]

def test_numberOfSubstrings():
    print(' --- Inside test_numberOfSubstrings ---')
    assert Solution().numberOfSubstrings("abcabc") == 10
    assert Solution().numberOfSubstrings("aaacb") == 3

def test_longestSubsequence():
    print(' --- Inside test_longestSubsequence ---')
    assert Solution().longestSubsequence("1001010", 5) == 5
    assert Solution().longestSubsequence("00101001", 1) == 6

def test_kthCharacter():
    print(' --- Inside test_kthCharacter ---')
    assert Solution().kthCharacter(5) == "b"
    assert Solution().kthCharacter(10) == "c"
