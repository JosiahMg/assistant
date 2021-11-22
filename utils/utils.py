
def is_equal(a, b, decimal=3):
    """比较两个结果是否相等
    """
    a = round(float(a), decimal)
    b = round(float(b), decimal)
    return a == b

