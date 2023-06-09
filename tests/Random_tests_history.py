# #gia test edw mesa
# def greet(name):
#     print ('Hello', name)
#
# def sum(x,y):
#     if type(x) == str :
#         if type(y) != str :
#             y= str(y)
#         print(x+y)
#     else :
#         print("Numbers")
#
# #greet('Jack')
# #greet('Jill')
# #greet('Bob')
#
# x = "PEPA"
# y = 6
# sum(x,y)
#
# import pandas as pd
# import numpy as np
#
# print("Hello World")
# print("It's a nice day")
# df = pd.DataFrame()
# var = np.nan
# #koo
#
# #####LAPTOP######
# # def my_func():
# #         '''this is a docstring'''
# #
# #     return 10//2
# #
# # #df = pd.DataFrame()
# # #var = np.nan
# # to=" ^.^ "
# # result=my_func()
# #
# # var=10 if isinstance(result, float) else 12
# # print(type(result), "eee", "ttototot", sep=to)
# # print(var)
# #print(max(100, 51, 14, key=lol ))
# # def lol(x):
# #     return ---x
# # my_list = [[2,3],[0,1],[4,5],[1,[1],0]]
# # print(my_list[1:], my_list, len(my_list), sorted(my_list), sep="\n")
# # help(max)
# # keys = {'Mercury': 'M',
# #  'Venus': 'V',
# #  'Earth': 'E',
# #  'Mars': 'M',
# #  'Jupiter': 'J',
# #  'Saturn': 'S',
# #  'Uranus': 'U',
# #  'Neptune': 'N'}
# #
# # for planet, initial in keys.items():
# #     print("{} begins with \"{}\"".format(planet.rjust(10), initial))
# def multi_word_search(doc_list, keywords):

class Node:
    def __init__(self, data, next):
        self.data = data
        self.next = next
        print("node data: " + str(self.data))
        print("node next: " + str(self.next))

######Provlima apo Sololearn sxetika me Linked Lists
class LinkedList:
    def __init__(self):
        self.head = None

    def add_at_front(self, data):
        self.head = Node(data, self.head)
        print("add: " + str(self.head))
    def add_at_end(self, data):
        if not self.head:
            self.head = Node(data, None)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(data, None)

    def get_last_node(self):
        n = self.head
        while (n.next != None):
            n = n.next
        return n.data

    def is_empty(self):
        return self.head == None

    def print_list(self):
        n = self.head
        while n != None:
            print(n.data, end=" => ")
            n = n.next
        print()


s = LinkedList()
s.add_at_front(5)
s.add_at_end(8)
s.add_at_front(9)

s.print_list()
print(s.get_last_node())