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
print(f'Move zeros - {move_zeroes([6,0,0,4,5])}')

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


def product_except_self(arr):
    n = len(arr)
    res = []
    for i in range(n):
        prod = 1
        for j in range(n):
            if j != i:
                prod *= arr[j]
        res.append(prod)
    return res

print(product_except_self([3, 5, 1, 2, 3, 6, 3]))

def product_except_self_optimal(arr):
    res = [1]*len(arr)
    prefix = 1
    for i in range(len(arr)):
        res[i] = prefix
        prefix *= arr[i]
    sufix = 1
    for i in range(len(arr)-1, -1, -1):
        res[i] *= sufix
        sufix *= arr[i]
    return res

print(product_except_self_optimal([3, 5, 1, 2, 3, 6, 3]))

def merge_sorted_array(arr1, arr2):
    l1 = len(arr1)
    l2 = len(arr2)
    l3 = l1+l2
    res = []
    i, j = 0, 0
    while i < l1 and j < l2:
        if arr1[i] < arr2[j]:
            res.append(arr1[i])
            i += 1
        else:
            res.append(arr2[j])
            j += 1
    while i < l1:
        res.append(arr1[i])
        i += 1
    while j < l2:
        res.append(arr2[j])
        j += 1
    return res

print(merge_sorted_array([0,3], [1,2]))

def merge_sorted_array_optimized(arr1, arr2):
    l1 = len(arr1)
    l2 = len(arr2)
    arr1.extend([0]*len(arr2))
    i, j, k = l1-1, l2-1, l1+l2-1
    while j >= 0:
        if i >= 0 and arr1[i] > arr2[j]:
            arr1[k] = arr1[i]
            i -= 1
        else:
            arr1[k] = arr2[j]
            j -= 1
        k -= 1
    return arr1

print(merge_sorted_array_optimized([ 1, 2, 3, 6,], [4,5,9]))

def rotate_arr(arr, k):
    n = len(arr)
    new_arr = []
    for i in range(n):
        new_arr.append(arr[(n-k+i)%n])
    return new_arr
print(rotate_arr([1,2,3,4,5,6,7,8,9], 3))
