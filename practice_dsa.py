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

#You have a list of words that you consider to be good and could be used for variable names.
# All the strings in words consist of lowercase English letters.
#A complex variable name is a combination (possibly with repetition) of some strings from words, written in CamelCase.
# In other words, all the strings are written without spaces, and each string (with the possible exception of the first one)
# starts with a capital letter.
words = ["is", "valid", "right"]
variableName = "isValid"
variableName1 = "isvalid"


def validate_variable(words, variableName):
    parts = []
    curr = ''
    for ch in variableName:
        if ch.isupper():
            parts.append(curr)
            curr = ch.lower()
        else:
            curr += ch
    if curr:
        parts.append(curr)

    for part in parts:
        if part not in words:
            return False
    return True


print(f'{variableName} is valid variable: {validate_variable(words, variableName)}')
print(f'{variableName1} is valid variable: {validate_variable(words, variableName1)}')

#Split String into Minimum Increasing Substrings
s = "ABCDEFFDEfghCBA"


#solution(s) = ["ABCDEF", "F", "DE", "fgh", "C", "B", "A"]
def split_string(s):
    res = []
    temp = s[0]
    for i in range(1, len(s)):
        if ord(s[i]) == ord(s[i - 1]) + 1:
            temp += s[i]
        else:
            res.append(temp)
            temp = s[i]
    if temp:
        res.append(temp)
    return res


print(f'{s} has split strings : {split_string(s)}')
