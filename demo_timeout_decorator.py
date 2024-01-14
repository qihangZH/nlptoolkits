from nlptoolkits._BasicKits.DecoratorT import timeout_process
import time


# Define the actual function in an imported module
@timeout_process(20)
def my_function():
    time.sleep(10)
    return "Function completed"


if __name__ == '__main__':
    try:
        result = my_function()
        print(result)
    except TimeoutError as e:
        print(e)
