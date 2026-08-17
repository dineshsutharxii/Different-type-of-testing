def two_sum(arr_, target):
    seen = {}
    for idx, val in enumerate(arr_):
        diff = target - val
        if diff in seen:
            return [seen[diff], idx]
        seen[val] = idx
    return []
print(two_sum([1,2,3,4,5], 7))

def contains_duplicate(arr):
    seen = set()
    for ele in arr:
        if ele in seen:
            return [ele, True]
        seen.add(ele)
    return []

print(contains_duplicate([1,2,5,4,5]))

def move_zeroes(arr):
    l = 0
    for r in range(len(arr)):
        if arr[r] != 0:
            arr[r], arr[l] = arr[l], arr[r]
            l += 1
    return arr
print(move_zeroes([6,0,0,4,5]))

def max_profit(arr):
    min_price = float('inf')
    max_pro = 0
    for ele in arr:
        min_price = min(min_price, ele)
        max_pro = max(max_pro, ele - min_price)
    return max_pro
print(max_profit([7, 1, 5, 3, 6, 4]))

def max_subarray(arr):
    max_sum = float('-inf')
    for i in range(len(arr)):
        current_sum  = 0
        for j in range(i, len(arr)):
            current_sum += arr[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum
print(max_subarray([-2, -1, 0, 1, 2]))

def max_subarray_optimized(arr):
    max_sum = float('-inf')
    current_sum = 0
    for i in range(len(arr)):
        if current_sum == 0:
            start = i
        current_sum += arr[i]
        if current_sum > max_sum:
            max_sum = current_sum
            ans_start, ans_end = start, i
        if current_sum < 0:
            current_sum = 0
    return [max_sum, arr[ans_start:ans_end]]

print(max_subarray_optimized([-2, -3, 4, -1, -2, 1, 5, -3]))

def missing_number(arr):
    n = len(arr)
    sum_ = n*(n+1)//2
    arr_sum = sum(arr)
    return sum_ - arr_sum

print(missing_number([3, 0, 1, 2, 5, 6, 7]))

def majority_element(arr):
    n = len(arr)
    ele_dict = {}
    for ele in arr:
        if ele in ele_dict:
            ele_dict[ele] += 1
        else:
            ele_dict[ele] = 1
    for val in ele_dict.values():
        if val > n//2:
            return val
    return 0

print(majority_element([3, 0, 1, 2, 3, 6, 3]))