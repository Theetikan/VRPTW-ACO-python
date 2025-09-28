import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue


class MultipleAntColonySystem:
    def __init__(self, graph: VrptwGraph, ants_num=10, beta=1, q0=0.1, whether_or_not_to_show_figure=True):
        super()
        # graph 结点的位置、服务时间信息
        self.graph = graph
        # ants_num 蚂蚁数量
        self.ants_num = ants_num
        # vehicle_capacity 表示每辆车的最大载重
        self.max_load = graph.vehicle_capacity
        # beta 启发性信息重要性
        self.beta = beta
        # q0 表示直接选择概率最大的下一点的概率
        self.q0 = q0
        # best path
        self.best_path_objective = None
        self.best_path = None
        self.best_vehicle_num = None
        self.best_metrics = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        轮盘赌
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob/sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def new_active_ant(ant: Ant, vehicle_num: int, local_search: bool, IN: np.ndarray, q0: float, beta: int, stop_event: Event):
        """
        按照指定的vehicle_num在地图上进行探索，所使用的vehicle num不能多于指定的数量，acs_time和acs_vehicle都会使用到这个方法
        对于acs_time来说，需要访问完所有的结点（路径是可行的），尽量找到travel distance更短的路径
        对于acs_vehicle来说，所使用的vehicle num会比当前所找到的best path所使用的车辆数少一辆，要使用更少的车辆，尽量去访问结点，如果访问完了所有的结点（路径是可行的），就将通知macs
        :param ant:
        :param vehicle_num:
        :param local_search:
        :param IN:
        :param q0:
        :param beta:
        :param stop_event:
        :return:
        """
        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])

        # 在new_active_ant中，最多可以使用vehicle_num个车，即最多可以包含vehicle_num+1个depot结点，由于出发结点用掉了一个，所以只剩下vehicle个depot
        unused_depot_count = vehicle_num

        # 如果还有未访问的结点，并且还可以回到depot中
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            # 计算所有满足载重等限制的下一个结点
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # 如果没有满足限制的下一个结点，则回到depot中
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # 开始计算满足限制的下一个结点，选择各个结点的概率
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains], ready_time)
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance-IN[next_index_meet_constrains])
            closeness = 1/distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][next_index_meet_constrains] * \
                              np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            # 按照概率直接选择closeness最大的结点
            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # 使用轮盘赌算法
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # 更新信息素矩阵
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # 如果走完所有的点了，需要回到depot
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # 对未访问的点进行插入，保证path是可行的
        ant.insertion_procedure(stop_event)

        # ant.index_to_visit_empty()==True就是feasible的意思
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                 global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        """
        ACS variant focused on minimizing total objective while respecting time windows.
        """

        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_objective = None
        global_best_metrics = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: new iteration')

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    True,
                    np.zeros(new_graph.node_num),
                    q0,
                    beta,
                    stop_event
                )
                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            ant_best_objective = None
            ant_best_path = None
            ant_best_ant = None

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: receive global path info')
                    global_best_path, global_best_objective, global_used_vehicle_num = info.get_path_info()
                    if hasattr(info, 'get_metrics'):
                        try:
                            global_best_metrics = info.get_metrics()
                        except TypeError:
                            global_best_metrics = None
                    else:
                        global_best_metrics = None

                if ant.index_to_visit_empty():
                    ant_objective = ant.total_objective
                    if ant_best_objective is None or ant_objective < ant_best_objective:
                        ant_best_objective = ant_objective
                        ant_best_path = list(ant.travel_path)
                        ant_best_ant = ant

            if global_best_path is not None and global_best_objective is not None:
                new_graph.global_update_pheromone(global_best_path, global_best_objective)

            if ant_best_objective is not None and (global_best_objective is None or ant_best_objective < global_best_objective):
                print("[acs_time]: ants' local search found an improved feasible path, send path info to macs")
                metrics = {}
                if ant_best_ant is not None:
                    metrics = {
                        'travel_time': ant_best_ant.total_travel_time,
                        'travel_distance': ant_best_ant.total_travel_distance,
                        'fixed_cost': ant_best_ant.total_fixed_cost,
                        'operational_cost': ant_best_ant.total_operational_cost,
                    }
                path_found_queue.put(PathMessage(ant_best_path, ant_best_objective, **metrics))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(new_graph: VrptwGraph, vehicle_num: int, ants_num: int, q0: float, beta: int,
                    global_path_queue: Queue, path_found_queue: Queue, stop_event: Event):
        """
        ACS variant focused on balancing vehicle usage while tracking total objective.
        """
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_objective = None
        global_best_metrics = None

        current_path, current_path_objective, _ = new_graph.nearest_neighbor_heuristic(max_vehicle_num=vehicle_num)
        current_metrics = getattr(new_graph, '_last_nnh_metrics', None)
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            if ind in current_index_to_visit:
                current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print('[acs_vehicle]: new iteration')

            if stop_event.is_set():
                print('[acs_vehicle]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    False,
                    IN,
                    q0,
                    beta,
                    stop_event
                )
                ants_thread.append(thread)
                ants.append(ant)

            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit] + 1

                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = list(ant.travel_path)
                    current_index_to_visit = list(ant.index_to_visit)
                    current_path_objective = ant.total_objective
                    current_metrics = {
                        'travel_time': ant.total_travel_time,
                        'travel_distance': ant.total_travel_distance,
                        'fixed_cost': ant.total_fixed_cost,
                        'operational_cost': ant.total_operational_cost,
                    }
                    IN = np.zeros(new_graph.node_num)

                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        path_found_queue.put(
                            PathMessage(
                                ant.travel_path,
                                ant.total_objective,
                                travel_time=ant.total_travel_time,
                                travel_distance=ant.total_travel_distance,
                                fixed_cost=ant.total_fixed_cost,
                                operational_cost=ant.total_operational_cost,
                            )
                        )

            if current_path is not None and current_path_objective is not None:
                new_graph.global_update_pheromone(current_path, current_path_objective)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                global_best_path, global_best_objective, global_used_vehicle_num = info.get_path_info()
                if hasattr(info, 'get_metrics'):
                    try:
                        global_best_metrics = info.get_metrics()
                    except TypeError:
                        global_best_metrics = None
                else:
                    global_best_metrics = None

            if global_best_path is not None and global_best_objective is not None:
                new_graph.global_update_pheromone(global_best_path, global_best_objective)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, file_to_write_path=None):
        """
        开启另外的线程来跑multiple_ant_colony_system， 使用主线程来绘图
        :return:
        """
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(target=self._multiple_ant_colony_system, args=(path_queue_for_figure, file_to_write_path, ))
        multiple_ant_colony_system_thread.start()

        # 是否要展示figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()

    def _multiple_ant_colony_system(self, path_queue_for_figure: MPQueue, file_to_write_path=None):
        """
        调用acs_time 和 acs_vehicle进行路径的探索
        :param path_queue_for_figure:
        :return:
        """
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, 'w')
        else:
            file_to_write = None

        start_time_total = time.time()

        # 在这里需要两个队列，time_what_to_do、vehicle_what_to_do， 用来告诉acs_time、acs_vehicle这两个线程，当前的best path是什么，或者让他们停止计算
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # 另外的一个队列， path_found_queue就是接收acs_time 和acs_vehicle计算出来的比best path还要好的feasible path
        path_found_queue = Queue()

        # 使用近邻点算法初始化
        self.best_path, self.best_path_objective, self.best_vehicle_num = self.graph.nearest_neighbor_heuristic()
        self.best_metrics = getattr(self.graph, '_last_nnh_metrics', None)
        initial_metrics = self.best_metrics or {}
        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_objective, **initial_metrics))

        while True:
            print('[multiple_ant_colony_system]: new iteration')
            start_time_found_improved_solution = time.time()

            # 当前best path的信息，放在queue中以通知acs_time和acs_vehicle当前的best_path是什么
            metrics_to_share = self.best_metrics or {}
            global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))
            global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))

            stop_event = Event()

            # acs_vehicle，尝试以self.best_vehicle_num-1辆车去探索，访问更多的结点
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num-1, self.ants_num, self.q0,
                                              self.beta, global_path_to_acs_vehicle, path_found_queue, stop_event))

            # acs_time 尝试以self.best_vehicle_num辆车去探索，找到更短的路径
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           global_path_to_acs_time, path_found_queue, stop_event))

            # 启动acs_vehicle_thread和acs_time_thread，当他们找到feasible、且是比best path好的路径时，就会发送到macs中来
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                # 如果在指定时间内没有搜索到更好的结果，则退出程序
                given_time = 10
                if time.time() - start_time_found_improved_solution > 60 * given_time:
                    stop_event.set()
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(file_to_write, 'time is up: cannot find a better solution in given time(%d minutes)' % given_time)
                    self.print_and_write_in_file(file_to_write, 'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total))
                    self.print_and_write_in_file(file_to_write, 'the best path have found is:')
                    self.print_and_write_in_file(file_to_write, self.best_path)
                    self.print_and_write_in_file(file_to_write, 'best path objective is %f, best vehicle_num is %d' % (self.best_path_objective, self.best_vehicle_num))
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    # 传入None作为结束标志
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(None, None))

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()
                    return

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: receive found path info')
                found_path, found_path_objective, found_path_used_vehicle_num = path_info.get_path_info()
                if hasattr(path_info, 'get_metrics'):
                    try:
                        found_path_metrics = path_info.get_metrics() or {}
                    except TypeError:
                        found_path_metrics = {}
                else:
                    found_path_metrics = {}

                while not path_found_queue.empty():
                    extra_info = path_found_queue.get()
                    extra_path, extra_objective, extra_vehicle_num = extra_info.get_path_info()
                    if hasattr(extra_info, 'get_metrics'):
                        try:
                            extra_metrics = extra_info.get_metrics() or {}
                        except TypeError:
                            extra_metrics = {}
                    else:
                        extra_metrics = {}

                    if extra_objective < found_path_objective:
                        found_path, found_path_objective, found_path_used_vehicle_num = extra_path, extra_objective, extra_vehicle_num
                        found_path_metrics = extra_metrics

                    elif extra_vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_objective, found_path_used_vehicle_num = extra_path, extra_objective, extra_vehicle_num
                        found_path_metrics = extra_metrics

                if found_path_objective < self.best_path_objective:

                    start_time_found_improved_solution = time.time()

                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        "[macs]: objective of found path (%f) better than best path's (%f)"
                        % (found_path_objective, self.best_path_objective)
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total)
                    )
                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_objective = found_path_objective
                    self.best_metrics = found_path_metrics

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_objective, **(self.best_metrics or {})))

                    metrics_to_share = self.best_metrics or {}
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))

                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    stop_event.set()

                if found_path_used_vehicle_num < best_vehicle_num:

                    start_time_found_improved_solution = time.time()

                    self.print_and_write_in_file(file_to_write, '*' * 50)
                    self.print_and_write_in_file(
                        file_to_write,
                        "[macs]: vehicle num of found path (%d) better than best path's (%d), found path objective is %f"
                        % (found_path_used_vehicle_num, best_vehicle_num, found_path_objective)
                    )
                    self.print_and_write_in_file(
                        file_to_write,
                        'it takes %0.3f second from multiple_ant_colony_system running' % (time.time()-start_time_total)
                    )
                    self.print_and_write_in_file(file_to_write, '*' * 50)

                    best_vehicle_num = found_path_used_vehicle_num
                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_objective = found_path_objective
                    self.best_metrics = found_path_metrics

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_objective, **(self.best_metrics or {})))

                    metrics_to_share = self.best_metrics or {}
                    global_path_to_acs_vehicle.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))
                    global_path_to_acs_time.put(PathMessage(self.best_path, self.best_path_objective, **metrics_to_share))


                # 如果，这两个线程找到的路径用的车辆更少了，就停止这两个线程，开始下一轮迭代

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message='default message'):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message)+'\n')
