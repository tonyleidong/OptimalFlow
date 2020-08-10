#!/usr/bin/env python

class autoFlow:
    def __init__(self,func = None):
        self.type = func
    def readlog(self,module_name = None):
        if module_name == "autoCV":
            file = open("./logs/autoCV_log_2020.08.07.23.23.41.log")
            lines = file.read().splitlines()
            file.close()
