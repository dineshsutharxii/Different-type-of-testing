import collections
from collections import deque
from heapq import heappop, heappush
from itertools import chain
from math import floor, log10


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
        if not p and not q:
            return True
        elif p and q and p.val == q.val:
            return self.identical_trees_or_not(p.left, q.left) and self.identical_trees_or_not(p.right, q.right)
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

    def subsetXORSum(self, nums):
        total = 0
        for num in nums:
            total |= num
        return total * (1 << (len(nums) - 1))

    def find_subset(self, arr):
        if not arr:
            return [[]]
        subsets = self.find_subset(arr[1:])
        return subsets + [[arr[0]] + s for s in subsets]

    def hourglassSum(self, arr):
        l = len(arr)
        max_sum = float('-inf')
        for i in range(l - 2):
            for j in range(l - 2):
                hourglass_sum = (arr[i][j] + arr[i][j + 1] + arr[i][j + 2] +
                                 arr[i + 1][j + 1] +
                                 arr[i + 2][j] + arr[i + 2][j + 1] + arr[i + 2][j + 2])
                max_sum = max(max_sum, hourglass_sum)
        return max_sum

    def minOperationsToMakeK(self, nums, k):
        if k > min(nums):
            return -1
        unique_elements = set(nums)
        unique_count = len(unique_elements)
        if k in unique_elements:
            return unique_count - 1
        else:
            return unique_count

    def maximumBeauty(self, items, queries):

        maxI = float('inf')
        res = [[0, 0, maxI]]
        items.sort(key=lambda x: x[0])
        for price, beauty in items:
            lastBracket = res[-1]
            if beauty > lastBracket[1]:
                res[-1][2] = price
                res.append([price, beauty, maxI])
        ans = []
        for x in queries:
            for i in range(len(res) - 1, -1, -1):
                if res[i][0] <= x:
                    ans.append(res[i][1])
                    break
        return ans

    def kthSmallest(self, root, k):
        res = []

        def inorder(node):
            if not node: return
            inorder(node.left)
            if len(res) == k:
                return
            res.append(node.val)
            inorder(node.right)

        inorder(root)
        return res[-1]

    def countPairs(self, nums, k) -> int:
        c = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] and (i * j) % k == 0:
                    c += 1
        return c

    def countAndSay(self, n: int) -> str:
        def find_count(s: str) -> str:
            result = []
            count = 1

            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    count += 1
                else:
                    result.append(f"{count}{s[i - 1]}")
                    count = 1

            result.append(f"{count}{s[-1]}")
            return "".join(result)

        result = "1"
        for _ in range(n - 1):
            result = find_count(result)
        return result

    def countFairPairs(self, nums, lower, upper):
        def count(t):
            i, j = 0, len(nums) - 1
            res = 0
            while i < j:
                if nums[i] + nums[j] > t:
                    j -= 1
                else:
                    res += j - i
                    i += 1
            return res

        nums.sort()
        return count(upper) - count(lower - 1)

    def numRabbits(self, answers):
        d = {}
        count = 0
        for i in answers:
            if i == 0:
                count += 1
            else:
                if i not in d or i == d[i]:
                    d[i] = 0
                    count += 1 + i
                else:
                    d[i] += 1
        return count

    def numberOfArrays(self, differences, lower, upper):
        mini_sum, maxi_sum, curr_sum = 0, 0, 0
        for diff in differences:
            curr_sum += diff
            mini_sum = min(mini_sum, curr_sum)
            maxi_sum = max(maxi_sum, curr_sum)
        seq_range = maxi_sum - mini_sum
        allowed_range = upper - lower
        if seq_range > allowed_range:
            return 0
        else:
            return allowed_range - seq_range + 1

    def maxMatrixSum(self, matrix):
        res = 0
        mini = float('inf')
        neg_cnt = 0
        for row in matrix:
            for val in row:
                res += abs(val)
                mini = min(mini, abs(val))
                if val < 0:
                    neg_cnt += 1
        if neg_cnt % 2 == 1:
            res -= 2 * mini
        return res

    def countLargestGroup(self, n):
        res = []
        for i in range(1, n + 1):
            res.append(sum(int(x) for x in str(i)))
        c = collections.Counter(res)
        x = [i for i in c.values() if i == max(c.values())]
        return len(x)

    def countCompleteSubarrays(self, nums):
        n = len(nums)
        target = len(set(nums))
        res = 0
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                if len(seen) == target:
                    res += n - j
                    break
        return res

    def countInterestingSubarrays(self, nums, modulo, k):
        ans = prefix = 0
        freq = collections.Counter({0: 1})
        for x in nums:
            if x % modulo == k: prefix += 1
            prefix %= modulo
            ans += freq[(prefix - k) % modulo]
            freq[prefix] += 1
        return ans

    def countSubarrays(self, nums, minK, maxK):
        ans = 0
        j = -1
        prevMinKIndex = -1
        prevMaxKIndex = -1
        for i, num in enumerate(nums):
            if num < minK or num > maxK:
                j = i
            if num == minK:
                prevMinKIndex = i
            if num == maxK:
                prevMaxKIndex = i
            ans += max(0, min(prevMinKIndex, prevMaxKIndex) - j)
        return ans

    def countSubarraysWithCondition(self, nums):
        cnt_of_subarray = 0
        for i in range(len(nums) - 2):
            if float(nums[i] + nums[i + 2]) == float(nums[i + 1] / 2):
                cnt_of_subarray += 1
        return cnt_of_subarray

    def countSubarraysWithScoreLessThanK(self, nums, k: int) -> int:
        def sum_and_max_multiplication(arr1):
            return sum(arr1) * (len(arr1))

        l, cnt = len(nums), 0
        subarray = []
        for i in range(l):
            for j in range(i, l):
                subarray.append(nums[i:j + 1])
        print(subarray)
        for ele in subarray:
            if sum_and_max_multiplication(ele) < k:
                cnt += 1
        return cnt

    def count_Subarrays_max_Element_Appears_at_Least_K_Times(self, nums, k):
        subarr = []
        n, cnt = len(nums), 0
        maxi = max(nums)
        for i in range(n):
            for j in range(i, n):
                if nums[i:j + 1] not in subarr:
                    subarr.append(nums[i:j + 1])

        def max_element_k_times(arr1, k):
            if arr1.count(maxi) >= k:
                return True

        for arr in subarr:
            if max_element_k_times(arr, k):
                cnt += 1
        return cnt

    def findCountOfNumbersWithEvenString(self, nums):
        # cnt = 0
        # for ele in nums:
        #     if len(str(ele))%2 == 0:
        #         cnt += 1
        # return cnt
        result = 0
        for num in nums:
            num_digits = floor(log10(num)) + 1
            if num_digits % 2 == 0:
                result += 1
        return result

    def addSpaces(self, s: str, spaces):
        m, n = len(spaces), len(s)
        t = [' '] * (m + n)
        j = 0
        for i, c in enumerate(s):
            if j < m and i == spaces[j]:
                j += 1
            t[i + j] = s[i]
        return "".join(t)

    def pushDominoes(self, dominoes: str) -> str:
        temp = ''
        while dominoes != temp:
            temp = dominoes
            dominoes = dominoes.replace('R.L', 'xxx')
            dominoes = dominoes.replace('R.', 'RR')
            dominoes = dominoes.replace('.L', 'LL')

        return dominoes.replace('xxx', 'R.L')

    def minDominoRotations(self, A, B) -> int:
        if len(A) != len(B): return -1
        same, countA, countB = collections.Counter(), collections.Counter(A), collections.Counter(B)
        for a, b in zip(A, B):
            if a == b:
                same[a] += 1
        for i in range(1, 7):
            if countA[i] + countB[i] - same[i] == len(A):
                return min(countA[i], countB[i]) - same[i]
        return -1

    def numTilings(self, n):
        dp = [1, 2, 5] + [0] * n
        for i in range(3, n):
            dp[i] = (dp[i - 1] * 2 + dp[i - 3]) % 1000000007
        return dp[n - 1]

    def buildArray(self, nums):
        ans = []
        for i in range(len(nums)):
            ans.append(nums[nums[i]])
        return ans

    def minTimeToReach(self, moveTime) -> int:
        onGrid = lambda x, y: 0 <= x < n and 0 <= y < m
        n, m, dx, dy = len(moveTime), len(moveTime[0]), 1, 0
        heap = [(0, 0, 0, 1)]
        seen = {(0, 0)}

        while heap:
            time, x, y, step = heappop(heap)
            if (x, y) == (n - 1, m - 1): return time
            for _ in range(4):
                X, Y, dx, dy = x + dx, y + dy, -dy, dx
                if onGrid(X, Y) and (X, Y) not in seen:
                    t = max(time, moveTime[X][Y]) + step
                    heappush(heap, [t, X, Y, 3 - step])
                    seen.add((X, Y))

    def maximumBeauty(self, nums, k):
        nums.sort()
        l = 0
        for r in range(len(nums)):
            if nums[r] - nums[l] > k * 2: l += 1
        return r - l + 1

    def minSum(self, nums1, nums2):
        count1 = nums1.count(0)
        sum1 = sum(nums1) + count1
        count2 = nums2.count(0)
        sum2 = sum(nums2) + count2
        if sum1 == sum2:
            return sum1
        if sum1 >= sum2:
            return sum1 if count2 else -1
        else:
            return sum2 if count1 else -1

    def threeConsecutiveOdds(self, arr):
        cnt = 0
        for ele in arr:
            if ele % 2 == 1:
                cnt += 1
            else:
                cnt = 0
            if cnt == 3: return True
        return False

    def findScore(self, nums):
        nums.append(float("inf"))
        res = 0
        start = -1
        i = 1
        while i < len(nums):
            if nums[i] >= nums[i - 1]:
                for j in range(i - 1, start, -2):
                    res += nums[j]
                start = i
                i += 1
            i += 1
        return res

    def lengthAfterTransformations(self, s, t):
        for i in range(t):
            new_str = ""
            for ele in s:
                if ele == 'z':
                    new_str += 'ab'
                else:
                    new_str += chr(ord(ele) + 1)
            s = new_str
        return len(new_str)

    def getFinalState(self, nums, k, multiplier: int):
        mini = 0
        for i in range(k):
            mini = min(nums)
            nums[nums.index(mini)] = mini * multiplier
        return nums

    def getLongestSubsequence(self, words, groups):
        res = [words[0]]
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                res.append(words[i])
        return res

    def finalPrices(self, prices):
        stk = []
        for i in range(len(prices)):
            while stk and (prices[stk[-1]] >= prices[i]):
                prices[stk.pop()] -= prices[i]
            stk.append(i)
        return prices

    def sortColors(self, nums) -> None:
        count_0, count_1, count_2 = 0, 0, 0
        for ele in nums:
            if ele == 0:
                count_0 += 1
            if ele == 1:
                count_1 += 1
            if ele == 2:
                count_2 += 1
        for i in range(count_0): nums[i] = 0
        for i in range(count_0, count_0 + count_1): nums[i] = 1
        for i in range(count_0 + count_1, count_0 + count_1 + count_2): nums[i] = 2
        return nums

    def maxChunksToSorted(self, arr):
        ans = 0
        prev_max = 0
        for idx, x in enumerate(arr):
            prev_max = max(prev_max, x)
            if prev_max == idx:
                ans += 1
        return ans

    def triangleType(self, nums):
        if not (nums[0] + nums[1] > nums[2] and nums[0] + nums[2] > nums[1] and nums[2] + nums[1] > nums[0]):
            return "none"
        if nums[0] == nums[1] == nums[2]:
            return "equilateral"
        elif (nums[0] == nums[1]) or (nums[1] == nums[2]) or (nums[0] == nums[2]):
            return "isosceles"
        else:
            return "scalene"

    def isZeroArray(self, nums, queries):
        n = len(nums)
        freq = [0] * n

        for q in queries:
            freq[q[0]] += 1
            if q[1] + 1 < n:
                freq[q[1] + 1] -= 1

        curFreq = 0
        for i in range(n):
            curFreq += freq[i]
            if curFreq < nums[i]:
                return False
        return True

    def setZeroes(self, matrix) -> None:
        m = len(matrix)
        n = len(matrix[0])
        arr1 = [float("-inf")] * m
        arr2 = [float("-inf")] * n
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    arr1[i] = arr2[j] = 0
        for i in range(m):
            for j in range(n):
                if arr1[i] == 0 or arr2[j] == 0:
                    matrix[i][j] = 0
        return matrix

    def maxRemoval(self, nums, queries):
        q = deque(sorted(queries))
        available = []
        working = []
        for i in range(len(nums)):
            while q and q[0][0] <= i:
                heappush(available, -q.popleft()[1])
            while working and working[0] < i:
                heappop(working)
            while nums[i] > len(working):
                if available and -available[0] >= i:
                    heappush(working, -heappop(available))
                else:
                    return -1
        return len(available)

    def findTargetSumWays(self, nums, target):
        counter = {0: 1}
        for n in nums:
            temp = {}
            for total, count in counter.items():
                temp[total + n] = temp.get(total + n, 0) + count
                temp[total - n] = temp.get(total - n, 0) + count
            counter = temp
        return counter.get(target, 0)

    def findWordsContaining(self, words, x):
        res = []
        for i in range(len(words)):
            if x in words[i]:
                res.append(i)
        return res

    def longestPalindrome(self, words):
        counter, ans = [[0] * 26 for _ in range(26)], 0
        for w in words:
            a, b = ord(w[0]) - ord('a'), ord(w[1]) - ord('a')
            if counter[b][a]:
                ans += 4
                counter[b][a] -= 1
            else:
                counter[a][b] += 1
        for i in range(26):
            if counter[i][i]:
                ans += 2
                break
        return ans

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        dp = [0] * (high + 1)
        dp[0] = 1
        md = 10 ** 9 + 7
        for i in range(min(zero, one), high + 1):
            if i >= zero:
                dp[i] = (dp[i] + dp[i - zero]) % md
            if i >= one:
                dp[i] = (dp[i] + dp[i - one]) % md

        s = 0
        for i in range(low, high + 1):
            s = (s + dp[i]) % md
        return s

    def maxCount(self, s):
        maxi = cnt_zero = 0
        cnt_one = s.count('1')
        for i in range(len(s) - 1):
            cnt_zero += s[i] == '0'
            cnt_one -= s[i] == '1'
            maxi = max(maxi, cnt_zero + cnt_one)
        return maxi

    def mincostTickets(self, days, costs) -> int:
        max_day = days[-1]
        memo = [0] * (max_day + 1)
        travel = [False] * (max_day + 1)
        memo[0] = 0
        for d in days:
            travel[d] = True
        for i in range(1, len(memo)):
            memo[i] = memo[i - 1]
            if travel[i]:
                memo[i] = min(memo[i - 1] + costs[0], memo[max(0, i - 7)] + costs[1], memo[max(0, i - 30)] + costs[2])
        return memo[-1]

    def waysToSplitArray(self, nums):
        leftSideSum = 0
        rightSideSum = sum(nums)
        validSplits = 0
        for i in range(len(nums) - 1):
            leftSideSum += nums[i]
            rightSideSum -= nums[i]
            if leftSideSum >= rightSideSum:
                validSplits += 1
        return validSplits

    def snakesAndLadders(self, board) -> int:
        board.reverse()
        for i in range(1, len(board), 2): board[i].reverse()
        arr = [None] + list(chain(*board))
        n, queue, seen, ct = len(arr) - 1, deque([1]), {1}, 0
        while queue:
            lenQ = len(queue)
            for _ in range(lenQ):
                cur = queue.popleft()
                if cur == n: return ct
                for i in range(cur + 1, min(cur + 7, n + 1)):
                    nxt = arr[i] if arr[i] + 1 else i
                    if nxt in seen: continue
                    seen.add(nxt)
                    queue.append(nxt)
            ct += 1
        return -1

    def distributeCandies(self, n, limit):
        res = 0
        for i in range(min(limit, n) + 1):
            if n - i <= 2 * limit:
                res += min(n - i, limit) - max(0, n - i - limit) + 1
        return res

    def findDifferentBinaryString(self, nums) -> str:
        def generate_binary_combinations(n, current="", result=None):
            if result is None:
                result = []
            if len(current) == n:
                result.append(current)
                return result
            generate_binary_combinations(n, current + "0", result)
            generate_binary_combinations(n, current + "1", result)
            return result

        combination = generate_binary_combinations(len(nums[0]))
        for ele in combination:
            if ele not in nums:
                return ele

    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        n = len(word)
        length = n - numFriends + 1
        maxi_c, res = max(word), ""

        for i in range(n):
            if word[i] == maxi_c:
                sub_str = word[i:i + length]
                res = max(res, sub_str)
        return res

    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        def find(c):
            i = ord(c) - ord('a')
            while root[i] != i:
                i = root[i]
            return i
        root = list(range(26))
        for c1, c2 in zip(s1,s2):
            r1 = find(c1)
            r2 = find(c2)
            if r1 > r2:
                r1, r2 = r2, r1
            root[r2] = r1
        return "".join(chr(ord('a') + find(c)) for c in baseStr)

    def clearStars(self, s: str) -> str:
        n = len(s)
        buckets = [[] for _ in range(26)]
        removed = [False] * n

        for i in range(n):
            if s[i] == '*':
                removed[i] = True
                for j in range(26):
                    if buckets[j]:
                        removed[buckets[j].pop()] = True
                        break
            else:
                buckets[ord(s[i]) - ord('a')].append(i)
        result = []
        for i in range(n):
            if not removed[i]:
                result.append(s[i])
        return ''.join(result)

    def lexicalOrder(self, n: int):
        res = []
        curr = 1
        for _ in range(n):
            res.append(curr)
            if curr * 10 <= n:
                curr *= 10
            else:
                while curr % 10 == 9 or curr + 1 > n:
                    curr //= 10
                curr += 1
        return res

    def canConstruct(self, s: str, k: int) -> bool:
        if k > len(s):
            return False
        h = collections.Counter(s)
        count_odd = 0
        for value in h.values():
            if value % 2:
                count_odd += 1
        if count_odd > k:
            return False
        return True

    def minimumLength(self, s: str) -> int:
        n = len(s)
        x = 0
        mp = {}
        for char in s:
            mp[char] = mp.get(char, 0) + 1
        for f in mp.values():
            x += (f - 1) if (f % 2 == 1) else (f - 2)
        return n - x

    def maxAdjacentDistance(self, nums) -> int:
        maxi = abs(nums[0] - nums[len(nums) - 1])
        for i in range(len(nums) - 1):
            maxi = max(maxi, abs(nums[i] - nums[i + 1]))
        return maxi