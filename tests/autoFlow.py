#!/usr/bin/env python

class autoFlow:
    def __init__(self,func = None):
        self.type = func
    def readlog(self,module_name = None):
        if module_name == "autoCV":
            file = open(./logs/)
            lines = file.read().splitlines()
            file.close()
