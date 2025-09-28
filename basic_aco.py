import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time


class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, beta=2, q0=0.1,
                 whether_or_not_to_show_figure=True, alpha=1.0):
        super()
        # graph 结点的位置、服务时间信息
        self.graph = graph
        # ants_num 蚂蚁数量
        self.ants_num = ants_num
        # max_iter 最大迭代次数
        self.max_iter = max_iter
        # vehicle_capacity 表示每辆车的最大载重
        self.max_load = graph.vehicle_capacity
        # beta 启发性信息重要性
        self.beta = beta
        self.alpha = alpha
        # q0 表示直接选择概率最大的下一点的概率
        self.q0 = q0
        # best path tracking
        self.best_path = None
        self.best_objective_value = None
        self.best_vehicle_num = None
        self.best_metrics = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure

    def run_basic_aco(self):
        # 开启一个线程来跑_basic_aco，使用主线程来绘图
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        # 是否要展示figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        # 传入None作为结束标志
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        """
        最基本的蚁群算法
        :return:
        """
        start_time_total = time.time()

        # 最大迭代次数
        start_iteration = 0
        for iter in range(self.max_iter):

            # 为每只蚂蚁设置当前车辆负载，当前旅行距离，当前时间
            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):

                # 蚂蚁需要访问完所有的客户
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    # 判断加入该位置后，是否还满足约束条件, 如果不满足，则再选择一次，然后再进行判断
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0

                    # 更新蚂蚁路径
                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                # 最终回到0位置
                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)

            # 计算所有蚂蚁的路径长度
            objective_values = np.array([ant.total_objective for ant in ants])

            best_index = np.argmin(objective_values)
            best_ant = ants[int(best_index)]
            best_objective = objective_values[best_index]

            if self.best_path is None or best_objective < self.best_objective_value:
                metrics = {
                    "travel_time": best_ant.total_travel_time,
                    "travel_distance": best_ant.total_travel_distance,
                    "fixed_cost": best_ant.total_fixed_cost,
                    "operational_cost": best_ant.total_operational_cost,
                }

                self.best_path = best_ant.travel_path
                self.best_objective_value = best_objective
                self.best_vehicle_num = self.best_path.count(0) - 1
                self.best_metrics = metrics
                start_iteration = iter

                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(
                        PathMessage(
                            self.best_path,
                            self.best_objective_value,
                            **metrics,
                        )
                    )

                print('\n')
                print(
                    '[iteration %d]: improved objective = %0.3f (time=%0.3f, distance=%0.3f, fixed=%0.3f, variable=%0.3f)'
                    % (
                        iter,
                        self.best_objective_value,
                        metrics["travel_time"],
                        metrics["travel_distance"],
                        metrics["fixed_cost"],
                        metrics["operational_cost"],
                    )
                )
                print('it takes %0.3f second Basic ACO running' % (time.time() - start_time_total))

            self.graph.global_update_pheromone(self.best_path, self.best_objective_value)

            given_iteration = 100
            if iter - start_iteration > given_iteration:
                print('\n')
                print('iteration exit: can not find better solution in %d iteration' % given_iteration)
                break

        print('\n')
        if self.best_metrics is None:
            print('no feasible path found')
        else:
            metrics = self.best_metrics
            print(
                'final best objective is %0.3f (time=%0.3f, distance=%0.3f, fixed=%0.3f, variable=%0.3f), number of vehicle is %d'
                % (
                    self.best_objective_value,
                    metrics["travel_time"],
                    metrics["travel_distance"],
                    metrics["fixed_cost"],
                    metrics["operational_cost"],
                    self.best_vehicle_num or 0,
                )
            )
        print('it takes %0.3f second ACO running' % (time.time() - start_time_total))

    def select_next_index(self, ant):
        """
        选择下一个结点
        :param ant:
        :return:
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        tau = np.power(
        self.graph.pheromone_mat[current_index][index_to_visit],
        self.alpha)
        
        eta_beta = np.power(
        self.graph.heuristic_info_mat[current_index][index_to_visit],
        self.beta)
        
        transition_prob = tau * eta_beta

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            # 使用轮盘赌算法
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)
        return next_index

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
