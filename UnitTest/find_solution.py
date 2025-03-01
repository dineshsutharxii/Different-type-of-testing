def twoSum(nums, target):
    h = {}
    for i, num in enumerate(nums):
        n = target - num
        if n not in h:
            h[num] = i
        else:
            return [h[n], i]


def minSubArrayLen(nums, target):
    left, sum_of_subarray, min_length = 0, 0, float('inf')
    for right in range(len(nums)):
        sum_of_subarray += nums[right]
        while sum_of_subarray >= target:
            min_length = min(min_length, right - left + 1)
            sum_of_subarray -= nums[left]
            left += 1
    if min_length == float('inf'):
        return 0
    return min_length


def balance_paranthesis(arr1):
    st = []
    for it in arr1:
        if it == '(' or it == '{' or it == '[':
            st.append(it)
        else:
            if len(st) == 0:
                return False
            ch = st[-1]
            st.pop()
            if (it == ')' and ch == '(') or (it == ']' and ch == '[') or (it == '}' and ch == '{'):
                continue
            else:
                return False
    return len(st) == 0


def longest_substring_without_repeating_characters(str1):
    res, longest = "", ""
    if len(str1) == 1: return 1
    for ele in str1:
        if ele not in res:
            res += ele
        else:
            res = res[res.index(ele) + 1:] + ele
        if len(res) > len(longest):
            longest = res
    return longest


def is_palindrome(str1="abcba"):
    l, r = 0, len(str1) - 1
    while l < r:
        if str1[l] != str1[r]:
            return False
        r -= 1
        l += 1
    return True


def find_second_largest(arr1=None):
    if arr1 is None:
        arr1 = [10, 6, 8, 4, 3, 1]
    largest = -1
    second_largest = -1
    for ele in arr1:
        if ele > largest:
            second_largest = largest
            largest = ele
        elif largest > ele > second_largest:
            second_largest = ele
    return second_largest


def find_nextGreaterElement(arr1, arr2):
    ans = []
    for ele in arr1:
        index_in_arr1 = arr2.index(ele)
        maxi, next_highest = ele, -1
        for j in range(index_in_arr1, len(arr2)):
            if maxi < arr2[j]:
                next_highest = arr2[j]
                break
        ans.append(next_highest)
    return ans


def longest_repeating_character_replacement(s: str, k: int):
    sub_arr = []
    maxlen = 0
    for i in range(len(s)):
        hash = [0] * 26
        maxfre = 0
        for j in range(i, len(s)):
            sub_arr.append(s[i:j])
            hash[ord(s[j]) - ord('A')] += 1
            maxfre = max(maxfre, hash[ord(s[j]) - ord('A')])
            changes = j - i + 1 - maxfre
            if changes <= k:
                maxlen = max(maxlen, j - i + 1)
            else:
                break
    return maxlen


def pow_x_n(x: float, n: int):
    #Brute force
    ans = 1.0
    if x == 0: return 0
    if n == 0: return 1
    for i in range(1, abs(n) + 1):
        ans *= x
    if n < 0: ans = 1 / ans
    return round(ans, 4)


def find_all_subsequences(nums):
    n = len(nums)
    ans = []
    for i in range(1 << n):
        sublist = []
        for j in range(n):
            if i & 1 << j:
                sublist.append(nums[j])
        ans.append(sublist)
    return ans


def check_if_power_of_two(n):
    if n == 1: return True
    if n % 2 == 1: return False
    while n%2 == 0:
        n //= 2
    return n == 1