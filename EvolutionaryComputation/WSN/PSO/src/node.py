
class Node:
    def __init__(self, idx, branches=None, isCH=False):
        self.idx=idx
        self.branches=branches if branches is not None else []
        self.isCH=isCH
        self.relay_data=0
        self.p_idx=-1


    def set_parent(self, parent_idx):
        self.p_idx=parent_idx