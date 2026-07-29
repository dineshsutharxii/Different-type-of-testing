#Find all the elements of list which contains word 'w'
from collections import Counter


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

#First Word from Scrambled Note
words = ["dog", "cat", "rat"]
note = "atgod"


#Output: dog

def first_word_with_counter(words, note):
    notes_count = Counter(note.lower())
    for word in words:
        if Counter(word.lower()) <= notes_count:
            return word
    return '-'


print(first_word_with_counter(words, note))


def first_word_without_counter(words, note):
    notes_count = {}
    for ch in note:
        if ch in notes_count:
            notes_count[ch] += 1
        else:
            notes_count[ch] = 1
    for word in words:
        temp = notes_count.copy()
        pssbl = True
        for ch in word:
            if temp.get(ch, 0) == 0:
                pssbl = False
                break
            temp[ch] -= 1
        if pssbl:
            return word
    return "-"


print(first_word_without_counter(words, note))


#The classic FizzBuzz interview question is:
#Write a program that prints numbers from 1 to N.
#However:
#If a number is divisible by 3, print "Fizz".
#If a number is divisible by 5, print "Buzz".
#If a number is divisible by both 3 and 5, print "FizzBuzz".
#Otherwise, print the number itself.

def finbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 5 == 0:
            print("Buzz")
        elif i % 3 == 0:
            print("Fizz")
        else:
            print(i)


finbuzz(30)

group = {
    5: "Buzz",
    3: "Fizz",
    7: "Jazz",
    11: "Rock"
}


def finbuzz_dynamic(N):
    for i in range(N + 1):
        res = ''
        for divisor, word in group.items():
            if i % divisor == 0:
                res += word
        print(res if res else i)


finbuzz_dynamic(30)


# 2.Given a string find the balance paranthesis set
# str_ = '[]{}(){(}'

def balanced_para(str):
    stk = []
    pair = {')': '(', '}': '{', ']': '['}
    for ele in str:
        if ele in '({[':
            stk.append(ele)
        elif ele in ')}]':
            if not stk or stk[-1] != pair[ele]:
                return False
            stk.pop()
    return len(stk) == 0


print(balanced_para('[]{}(){(}'))
print(balanced_para('[]{}()()'))


def balanced_pairs(str):
    stk = []
    pairs = []
    pair = {')': '(', '}': '{', ']': '['}
    for ele in str:
        if ele in '({[':
            stk.append(ele)
        elif ele in ')}]':
            if stk and stk[-1] == pair[ele]:
                pairs.append(stk.pop() + ele)
    return pairs


print(balanced_pairs('[]{}()()'))
print(balanced_pairs('([{}()()])'))


#3.Given a string found the sum of integers found
def sum_of_integer(str):
    sum_ = 0
    for ele in str:
        if ord('0') <= ord(ele) <= ord('9'):
            sum_ += int(ele)
    return sum_


print(sum_of_integer('hjksh676sjnbkjsdffu'))


# 4.Given a string found the largest occurrence of consecutive 1’s
def find_largest_con_1(str):
    maxi_len = 0
    curr_len = 0
    for i in range(len(str)):
        if str[i] == '1':
            curr_len += 1
        else:
            maxi_len = curr_len if curr_len > maxi_len else maxi_len
            curr_len = 0
    return curr_len if curr_len > maxi_len else maxi_len


print(find_largest_con_1('1111011111100011111111111'))

#Find the length of the longest substring without repeating characters
strin = "abbbccccdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def find_largest_substring_without_repeating(str):
    maxi_len = 0
    curr_str = ''
    for ele in str:
        if ele not in curr_str:
            curr_str += ele
        else:
            maxi_len = max(maxi_len, len(curr_str))
            curr_str = curr_str[curr_str.index(ele) + 1:] + ele
    return max(maxi_len, len(curr_str))


print(find_largest_substring_without_repeating(strin))

def find_largest_substring_without_repeating_optimized(s):
    left = 0
    max_len = 0
    seen = {}
    longest = ''

    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        curr_str = s[left:right + 1]

        if len(curr_str) > max_len:
            max_len = len(curr_str)
            longest = curr_str
    return longest, max_len

print(find_largest_substring_without_repeating_optimized(strin))

#Given an array of non-negative integers, the task is to find the minimum number of elements such that their sum
#should be greater than the sum of the rest of the elements of the array.
#Input: arr[] = [3 , 1 , 7, 1 ]
#Output: 1
#Input:  arr[] = [2 , 1 , 2 ]
#Output: 2

arr1 = [5, 7, 1, 9, 3, 10, 16, 10, 8]
arr1 = sorted(arr1)[::-1]
sum_ = sum(arr1)
curr_sum = 0
count = 0
for ele in arr1:
    if sum_ >= curr_sum:
        curr_sum += ele
        sum_ -= ele
        count += 1
        print(f'curr_sum :{curr_sum}, sum_ : {sum_}, count_ : {count}')
print(count)


#Move all zeros to the end without using extra space.
def move_zeros(arr_):
    temp_arr = []
    for ele in arr_:
        if ele != 0:
            temp_arr.append(ele)
    while len(temp_arr) < len(arr_):
        temp_arr.append(0)
    return temp_arr
print(move_zeros([1,0,11,0,5]))

def move_zeros_optimized(arr_):
    write = 0
    for read in range(len(arr_)):
        if arr_[read] != 0:
            arr_[read], arr_[write] = arr_[write], arr_[read]
            write += 1
    return arr_
print(move_zeros_optimized([1,0,11,0,5]))

def two_sum(nums,target):
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None
print(two_sum([2,7,11,15], 26))

def two_sum_optimized(nums,target):
    p = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in p:
            return [p[diff], i]
        p[num] = i
    return None
print(two_sum_optimized([2,7,11,15], 18))


def palindrome(str):
    # return str == str[::-1]
    for i in range(len(str)):
        if str[i] != str[len(str) - 1 - i]:
            return False
    return True
print(palindrome('abcba'))