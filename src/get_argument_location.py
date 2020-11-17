#======================================================================================================
#The following three functions are used for get the argument of locations of the initial configurations
#======================================================================================================
def list_only_dir(directory):
    """This function list the directorys only under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    full_li = []
    #directory = '../data'
    for item in list_dir:
        li = [directory,item]
        full_li.append("/".join(li))
    return full_li
def list_only_naked_dir(directory):
    """This function list the naked directory names only under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    for item in list_dir:
        li = [directory,item]
    return list_dir
def str2int(list_dir):
    res = []
    for item in range(len(list_dir)):
        res.append(int(list_dir[item]))
    return res

if __name__ == "__main__":
    data_list = list_only_naked_dir('../data')
    print(str2int(data_list))
    print(data_list)
