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