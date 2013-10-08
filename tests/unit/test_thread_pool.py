"""
Created on Oct 8, 2013

Unit test for ThreadPool().

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import unittest
import thread_pool
import numpy.random
import time
import threading


class TestThreadPool(unittest.TestCase):
    def job(self, n_jobs, data_lock):
        time.sleep(numpy.random.rand() * 2 + 1)
        data_lock.acquire()
        n_jobs[0] -= 1
        data_lock.release()

    def test(self):
        print("Will test ThreadPool with 32 max threads.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(max_free_threads=32, max_threads=32,
                                      max_enqueued_tasks=32)
        n = 100
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self.job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(n_jobs[0], 0,
            "ThreadPool::shutdown(execute_remaining=True) is not working "
            "as expected.")

    def test2(self):
        print("Will test ThreadPool with 320 max threads.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(max_free_threads=32, max_threads=320,
                                      max_enqueued_tasks=320)
        n = 100
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self.job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(n_jobs[0], 0,
            "ThreadPool::shutdown(execute_remaining=True) is not working "
            "as expected.")

    def test3(self):
        print("Will test ThreadPool with max_free_threads=0.")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(max_free_threads=0, max_threads=32,
                                      max_enqueued_tasks=32)
        n = 10
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self.job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=False)
        self.assertEqual(n_jobs[0], 10,
            "ThreadPool::shutdown(execute_remaining=False) is not working "
            "as expected.")

    def test4(self):
        print("Will test ThreadPool for double shutdown().")
        n_jobs = [0]
        data_lock = threading.Lock()
        pool = thread_pool.ThreadPool(max_free_threads=1, max_threads=32,
                                      max_enqueued_tasks=32)
        n = 10
        for i in range(n):
            data_lock.acquire()
            n_jobs[0] += 1
            data_lock.release()
            pool.request(self.job, (n_jobs, data_lock))
        pool.shutdown(execute_remaining=True)
        self.assertEqual(n_jobs[0], 0,
            "ThreadPool::shutdown(execute_remaining=True) is not working "
            "as expected.")
        pool.shutdown(execute_remaining=True)
        self.assertEqual(n_jobs[0], 0,
            "ThreadPool::shutdown(execute_remaining=True) is not working "
            "as expected.")

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test']
    unittest.main()
