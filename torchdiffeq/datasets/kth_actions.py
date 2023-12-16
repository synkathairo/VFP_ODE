import deeplake 


def kth_actions(): 
    ds = deeplake.load("hub://activeloop/kth-actions")
    return ds 
