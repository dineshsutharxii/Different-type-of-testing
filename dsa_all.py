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