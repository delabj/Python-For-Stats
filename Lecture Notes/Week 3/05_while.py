# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:31:41 2019
@author: haritima
In this script we will learn how to implement While loop

Syntax of while loop

while (condition):
    # execute a statment 
"""

#%% Infinite loop 
a = 0
while a<10:
    print('current value ', a) 

#%% Simple while loop with an increment 
a = 0
while a < 4:
     print('current value', a) 
     a +=1 #a = a + 1 

#%% 
b = 20 
while b > 15:
    print('current value', b) 
    b -= 1 #Called AUGMENTED ASSIGNMENT
    
#%%
b = 10 
while b < 15:
    print('current value', b) 
    b += 1
else: 
    print('Phew! finally done.')
    

