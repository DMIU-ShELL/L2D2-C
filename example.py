# PIPE COMMUNICATION

import os
import re
from multiprocessing import Pipe
from os.path import join

import time
from multiprocessing.context import Process

WIND_REGEX = "\d* METAR.*EGLL \d*Z [A-Z ]*(\d{5}KT|VRB\d{2}KT).*="
WIND_EX_REGEX = "(\d{5}KT|VRB\d{2}KT)"
VARIABLE_WIND_REGEX = ".*VRB\d{2}KT"
VALID_WIND_REGEX = "\d{5}KT"
WIND_DIR_ONLY_REGEX = "(\d{3})\d{2}KT"
TAF_REGEX = ".*TAF.*"
COMMENT_REGEX = "\w*#.*"
METAR_CLOSE_REGEX = ".*="


def parse_to_array(text_conn, metars_conn):
    text = text_conn.recv()
    while text is not None:
        lines = text.splitlines()
        metar_str = ""
        metars = []
        for line in lines:
            if re.search(TAF_REGEX, line):
                break
            if not re.search(COMMENT_REGEX, line):
                metar_str += line.strip()
            if re.search(METAR_CLOSE_REGEX, line):
                metars.append(metar_str)
                metar_str = ""
        metars_conn.send(metars)
        text = text_conn.recv()
    metars_conn.send(None)


def extract_wind_direction(metars_conn, winds_conn):
    metars = metars_conn.recv()
    while metars is not None:
        winds = []
        for metar in metars:
            if re.search(WIND_REGEX, metar):
                for token in metar.split():
                    if re.match(WIND_EX_REGEX, token): winds.append(re.match(WIND_EX_REGEX, token).group(1))
        winds_conn.send(winds)
        metars = metars_conn.recv()
    winds_conn.send(None)


def mine_wind_distribution(winds_conn, wind_dist_conn):
    wind_dist = [0] * 8
    winds = winds_conn.recv()
    while winds is not None:
        for wind in winds:
            if re.search(VARIABLE_WIND_REGEX, wind):
                for i in range(8):
                    wind_dist[i] += 1
            elif re.search(VALID_WIND_REGEX, wind):
                d = int(re.match(WIND_DIR_ONLY_REGEX, wind).group(1))
                dir_index = round(d / 45.0) % 8
                wind_dist[dir_index] += 1
        winds = winds_conn.recv()
    wind_dist_conn.send(wind_dist)


if __name__ == '__main__':
    text_conn_a, text_conn_b = Pipe()
    metars_conn_a, metars_conn_b = Pipe()
    winds_conn_a, winds_conn_b = Pipe()
    winds_dist_conn_a, winds_dist_conn_b = Pipe()
    Process(target=parse_to_array, args=(text_conn_b, metars_conn_a)).start()
    Process(target=extract_wind_direction, args=(metars_conn_b, winds_conn_a)).start()
    Process(target=mine_wind_distribution, args=(winds_conn_b, winds_dist_conn_a)).start()
    #path_with_files = "../metarfiles"
    #start = time.time()
    #for file in os.listdir(path_with_files):
    #    f = open(join(path_with_files, file), "r")
    #    text = f.read()
    for i in range(50):
        text_conn_a.send(text)
    text_conn_a.send(None)
    wind_dist = winds_dist_conn_b.recv()
    end = time.time()
    print(wind_dist)
    print("Time taken", end - start)









# QUEUE POOLING

import re
import time
from multiprocessing import Process, Queue
# (45,11),(41,15),(36,20)

PTS_REGEX = "\((\d*),(\d*)\)"
TOTAL_PROCESSES = 8


def find_area(points_queue):
    points_str = points_queue.get()
    while points_str is not None:
        points = []
        area = 0.0
        for xy in re.finditer(PTS_REGEX, points_str):
            points.append((int(xy.group(1)), int(xy.group(2))))

        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            area += a[0] * b[1] - a[1] * b[0]
        area = abs(area) / 2.0
        # print(area)
        points_str = points_queue.get()


if __name__ == '__main__':
    queue = Queue(maxsize=1000)
    processes = []
    for i in range(TOTAL_PROCESSES):
        p = Process(target=find_area, args=(queue,))
        processes.append(p)
        p.start()
    f = open("polygons.txt", "r")
    lines = f.read().splitlines()
    start = time.time()
    for line in lines:
        queue.put(line)
    for _ in range(TOTAL_PROCESSES): queue.put(None)
    for p in processes: p.join()
    end = time.time()
    print("Time taken", end - start)









from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception


def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            print(task)
            tasks_that_are_done.put(task + ' is done by ' + current_process().name)
            time.sleep(.5)
    return True


def main():
    number_of_task = 10
    number_of_processes = 4
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for i in range(number_of_task):
        tasks_to_accomplish.put("Task no " + str(i))

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()