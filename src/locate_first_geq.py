def locate_first_geq(array, target) -> int:
    """
        Find index of the FIRST element >= target in a sorted array
    :param array:
    :param target:
    :return:
    """
    left, right = 0, len(array)
    mid = (left + right)//2

    while left < right:
        if array[mid] >= target:
            right = mid
            mid = (left + right)//2
            # Case 2: mid is NOT the first larger item
        elif array[mid] < target:
            left = mid+1
            mid = (left + right)//2

    return -1 if mid >= len(array) else mid