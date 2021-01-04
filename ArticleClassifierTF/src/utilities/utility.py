from typing import List


# Removes the enclosed lists that are empty.
def remove_empty_lists(list_of_lists: List[List[str]]):
    return list(filter(lambda list: len(list) > 0, list_of_lists))

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3