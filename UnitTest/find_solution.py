from collections import deque


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
    while n % 2 == 0:
        n //= 2
    return n == 1


def divide_two_integers(dividend: int, divisor: int):
    if dividend == -2 ** 31 and divisor == -1:
        return 2 ** 31 - 1
    if dividend == -2 ** 31 and divisor == 1:
        return -2 ** 31
    negative = (dividend < 0) ^ (divisor < 0)
    absDividend, absDivisor = abs(dividend), abs(divisor)
    quotient = 0
    while absDividend >= absDivisor:
        tempDivisor, multiple = absDivisor, 1
        while absDividend >= (tempDivisor << 1):
            tempDivisor <<= 1
            multiple <<= 1
        absDividend -= tempDivisor
        quotient += multiple

    return -quotient if negative else quotient


def find_single_num(nums):
    #using dict
    # freq = {}
    # for ele in nums:
    #     if ele not in freq:
    #         freq[ele] = 1
    #     else:
    #         freq[ele] += 1
    # for key in freq.keys():
    #     if freq[key] == 1:
    #         return key
    # return 0

    #Using XOR
    one_ele = 0
    for ele in nums:
        one_ele = one_ele ^ ele
    return one_ele


def count_primes(n):
    prime = []
    for i in range(n + 1):
        prime.append(1)
    p = 2
    while (p * p <= n):
        if prime[p] == 1:
            for i in range(p * p, n + 1, p):
                prime[i] = 0
        p += 1
    primecount = 0
    for i in range(2, n):
        if prime[i]:
            primecount += 1
    return primecount


def reverse_integer(x):
    res = 0
    res = (int(str(x)[1:][::-1])) * -1 if x < 0 else int(str(x)[::-1])
    if res > 2 ** 31 - 1 or res < -2 ** 31: return 0
    return res


def find_missing_and_repeated_number(grid):
    l = len(grid)
    missing = (l * l) * (l * l + 1) // 2
    seen_set, duplicate = set(), -1
    for row in grid:
        for num in row:
            if num not in seen_set:
                seen_set.add(num)
                missing -= num
            else:
                duplicate = num
    return [duplicate, missing]


def assign_cookies(g, s):
    g.sort()
    s.sort()
    left, right, count = 0, 0, 0
    while left < len(g) and right < len(s):
        if g[left] <= s[right]:
            count += 1
            left += 1
        right += 1
    return count


def lemonado_change(bills):
    five, ten = 0, 0
    for ele in bills:
        if ele == 5:
            five += 1
        elif ele == 10:
            if five:
                ten += 1
                five -= 1
            else:
                return False
        else:
            if ten and five:
                five -= 1
                ten -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    return True


def valid_parenthesis_string(s):
    count_str = 0
    stk = []
    for ele in s:
        if ele == "(":
            stk.append(ele)
        elif len(stk) > 0 and ele == ")" and stk[-1] == "(":
            stk.pop()
        else:
            count_str += 1
    if len(stk) == 0: return True
    if count_str == len(stk):
        return True
    return False


def jump_game(nums):
    max_length = 0
    for i in range(len(nums)):
        if i > max_length: return False
        if max_length < i + nums[i]:
            max_length = i + nums[i]
    return True


def jump_game_2(nums):
    jumps, l, r, n = 0, 0, 0, len(nums)
    while r < n - 1:
        farthest = 0
        for i in range(l, r + 1):
            farthest = max(farthest, i + nums[i])
        l = r + 1
        r = farthest
        jumps += 1
    return jumps


def insert_interval(intervals, newInterval):
    res = []
    for i in range(len(intervals)):
        if newInterval[1] < intervals[i][0]:
            res.append(newInterval)
            return res + intervals[i:]
        elif newInterval[0] > intervals[i][1]:
            res.append(intervals[i])
        else:
            newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
    res.append(newInterval)
    return res


def merge_intervals(intervals):
    merged = []
    intervals.sort(key=lambda x: x[0])
    prev = intervals[0]
    for interval in intervals[1:]:
        if interval[0] <= prev[1]:
            prev[1] = max(prev[1], interval[1])
        else:
            merged.append(prev)
            prev = interval
    merged.append(prev)
    return merged


def non_overlapping_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count, prev, l = 1, 0, len(intervals)
    for i in range(1, l):
        if intervals[i][0] >= intervals[prev][1]:
            count += 1
            prev = i
    return l - count


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def postorderTraversal(self, root):
        res = []
        if not root: return res
        res += self.postorderTraversal(root.left)
        res += self.postorderTraversal(root.right)
        res += [root.val]
        return res

    def preorderTraversal(self, root):
        res = []
        if not root: return res
        res += [root.val]
        res += self.preorderTraversal(root.left)
        res += self.preorderTraversal(root.right)
        return res

    def inorderTraversal(self, root):
        res = []
        if not root: return res
        res += self.inorderTraversal(root.left)
        res += [root.val]
        res += self.inorderTraversal(root.right)
        return res

    def levelorderTraversal(self, root):
        res = []
        queue = [root]
        if root == None:
            return res
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res

    def depth_of_binary_tree(self, root):
        if not root: return 0
        left_subtree = self.depth_of_binary_tree(root.left)
        right_subtree = self.depth_of_binary_tree(root.right)
        return max(left_subtree, right_subtree) + 1

    def balanced_binary_tree(self, root):
        return self.height(root) != -1

    def height(self, root):
        if root == None: return 0
        lh = self.height(root.left)
        rh = self.height(root.right)
        if rh == -1 or lh == -1: return -1

        if abs(lh - rh) > 1: return -1
        return max(lh, rh) + 1

    def diameter_of_binary_tree(self, root):
        def diameter(node, res):
            if not node:
                return 0
            lh = diameter(node.left, res)
            rh = diameter(node.right, res)

            res[0] = max(res[0], lh + rh)
            return max(lh, rh) + 1

        res = [0]
        diameter(root, res)
        return res[0]

    def identical_trees_or_not(self, p, q):
        if not p and not q: return True
        elif p and q and p.val == q.val: return self.identical_trees_or_not(p.left, q.left) and self.identical_trees_or_not(p.right, q.right)
        else:
            return False

    def zigzagLevelOrder(self, root):
        if not root: return []
        queue = deque([root])
        result, direction = [], 1
        while queue:
            level = [node.val for node in queue]
            if direction == -1:
                level.reverse()
            result.append(level)
            direction *= -1
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result

    def search_in_bst(self, root, val):
        stk = [root]
        if stk:
            node = stk.pop()
            print(node)
            if node.val == val:
                return node
            elif node.val < val and node.right:
                stk.append(node.right)
            elif node.val > val and node.left:
                stk.append(node.left)
        return None

    def max_sum_path(self, arr1, arr2):
        i = j = 0
        len_arr1, len_arr2 = len(arr1), len(arr2)
        sum_arr1 = sum_arr2 = res = 0
        while i < len_arr1 and j < len_arr2:
            if arr1[i] < arr2[j]:
                sum_arr1 += arr1[i]
                i += 1
            elif arr2[j] < arr1[i]:
                sum_arr2 += arr2[j]
                j += 1
            else:
                res += max(sum_arr1, sum_arr2) + arr1[i]
                i += 1
                j += 1
                sum_arr1 = sum_arr2 = 0
        while i < len_arr1:
            sum_arr1 += arr1[i]
            i += 1
        while j < len_arr2:
            sum_arr2 += arr2[j]
            j += 1
        res += max(sum_arr1, sum_arr2)
        return res
