
from pathos.multiprocessing import ProcessingPool

class Bar:
    def foo(self, name):
        return len(str(name))
    def boo(self, things):
        for thing in things:
            self.sum += self.foo(thing)
        return self.sum
    sum = 0


class Stick:
    def foo(self, name):
        return len(str(name))
    def boo(self, things):
        for thing in things:
            self.sum += self.foo(thing)
        return self.sum
    sum = 0


results = [[12,3,456], [8,9,10], ['a','b','cde'], [12,3,456], [8,9,10], ['a','b','cde'], [12,3,456], [8,9,10], ['a','b','cde'], [12,3,456], [8,9,10], ['a','b','cde'], [12,3,456], [8,9,10], ['a','b','cde']]
while True:
    b = Bar()
    p = ProcessingPool(2)

    results = p.map(b.boo, results)
    print(results)
    if len(results) == 1:
        break
    









'''

import os
from multiprocessing import Pool

import dill


def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


if __name__ == "__main__":

    pool = Pool(processes=5)

    # asyn execution of lambda
    jobs = []
    for i in range(10):
        job = apply_async(pool, lambda a, b: (a, b, a * b), (i, i + 1))
        jobs.append(job)

    for job in jobs:
        print(job.get())
    print

    # async execution of static method

    class O(object):

        @staticmethod
        def calc():
            return os.getpid()

    jobs = []
    for i in range(10):
        job = apply_async(pool, O.calc, ())
        jobs.append(job)

    for job in jobs:
        print (job.get())
'''