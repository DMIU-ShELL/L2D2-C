import multiprocessing

workers = (multiprocessing.cpu_count() * 2 + 1)
timeout = 2
compress = True