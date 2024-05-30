def reserved(func):
    reserved = False

    def inner(*args, **kwargs):
        nonlocal reserved
        if reserved:
            return 'В данный момент бот занят, пожалуйста, поробуйте позже!'
        reserved = True

        res = func(*args, **kwargs)
        reserved = False
        return res
    return inner 
