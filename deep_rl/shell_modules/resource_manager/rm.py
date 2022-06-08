# -*- coding: utf-8 -*-

class ResourceManager:
    OP_ID_DETECT = 0
    OP_ID_COMMS = 1
    OP_ID_TRAIN = 2
    OP_ID_EVAL = 3

    def __init__(self):
        return

    def operation(self, op_id):
        # determine whether to run an operation based on
        # system workload and priority

        # not yet implemented
        # always return True for now, running all operations
        status = True
        return status 

    def swap_b(self):
        return
