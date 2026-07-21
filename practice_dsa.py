#Find all the elements of list which contains word 'w'
def find_elements_in_list_which_contains_word(listy, w):
    for ele in listy:
        l_ptr, w_ptr = 0, 0
        while l_ptr < len(ele) and w_ptr < len(w):
            if w[w_ptr] == ele[l_ptr]:
                w_ptr += 1
                l_ptr += 1
            else:
                l_ptr += 1
        if w_ptr == len(w):
            print(f"{w} is present in {ele}")


listy = ['sfskwomfjprifse', 'koomprise', 'jkomprisxmsl', 'saxuuywq', 'komprise']
w = "komprise"
find_elements_in_list_which_contains_word(listy, w)


def bin_search(_arr, ele):
    l, u = 0, len(_arr) - 1
    while l <= u:
        m = (l + u) // 2
        if ele == _arr[m]:
            return True
        if ele > _arr[m]:
            l = m + 1
        else:
            u = m - 1
    return False


_arr = [1, 5, 6, 9, 11, 12, 18]
ele, ele1 = 18, 15
print(f'is {ele} present : ', bin_search(_arr, ele))
print(f'is {ele1} present : ', bin_search(_arr, ele1))


def bubble_sort(arr):
    for i in range(len(arr) - 1):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


arr = [6, 7, 2, 7, 2, 6, 1, 4, 60]
print(bubble_sort(arr))
