# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:25:57 2019

@author: Asgard
"""

import os, sys
 
class OuterClassA(object):
    
    def __init__(self):
        
        self.a=0
#        self.i=self.InnerClass()
    
    def outer_func_1(self, text):
        
        print(text)
    
    class InnerClass(object):
        
        def __init__(self):
            
            self.out = OuterClassA()
            
        def inner_func_1(self):
            
            self.out.outer_func_1('from inside')

outer = OuterClassA()
outer.outer_func_1('outside')
outer.InnerClass().inner_func_1()

