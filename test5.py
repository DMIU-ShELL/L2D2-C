from memory_profiler import profile
import time

class MyClass:
    def __init__(self):
        self.data = [0] * 10**6  # Creating a large list in the constructor

    @profile
    def process_data(self):
        result = sum(self.data)  # Accessing the large list
        return result

if __name__ == "__main__":
    obj = MyClass()

    while True:
        time.sleep(1)
        obj.process_data()