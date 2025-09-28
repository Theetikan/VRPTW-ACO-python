#from curses import raw
import numpy as np
import copy
from typing import List, Dict, Optional



class Node:
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


#from curses import raw
import numpy as np
import copy
from typing import List, Dict, Optional

from pyparsing import line


class Node:
    def __init__(self, id:  int, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class VrptwGraph:
    def __init__(self, file_path: str, vehicle_file: Optional[str] = None, rho=0.1):
        super()
        # 1) โหลด nodes/ระยะทาง
        self.node_num, self.nodes, self.node_dist_mat = self.create_from_file(file_path)

        # 2) โหลดข้อมูลรถให้เรียบร้อย "ก่อน" ใช้ NN heuristic
        self.vehicle_types = []
        self.vehicle_fleet = []
        self.vehicle_capacities = []
        self.vehicle_num = 0
        self.vehicle_capacity = None

        if vehicle_file is not None:
            self.vehicle_types = self._create_vehicle_types_from_file(vehicle_file)
            self.vehicle_fleet = self._expand_vehicle_fleet(self.vehicle_types)
            self.vehicle_capacities = [v.get("capacity", 0) for v in self.vehicle_fleet]
            self.vehicle_num = len(self.vehicle_fleet)
            if self.vehicle_capacities:
                self.vehicle_capacity = max(self.vehicle_capacities)
        else:
            print("Warning: No vehicle file")

        if self.vehicle_capacity is None:
            self.vehicle_capacity = float('inf')
        self._fleet_warning_issued = False
        #     # อ่านจากไฟล์ instance ฟอร์แมตเดิม (บรรทัดที่ 5)
        #     self.vehicle_num, self.vehicle_capacity = self._read_legacy_vehicle_info(file_path)
        #     self.vehicle_types = [{
        #         "type_id": 0,
        #         "num": self.vehicle_num,
        #         "capacity": float(self.vehicle_capacity),
        #         "fixed_cost": 0.0,
        #         "var_cost": 1.0
        #     }]

        # 3) ค่าที่เหลือ
        self.rho = rho
        self.nnh_travel_path, nnh_objective, _ = self.nearest_neighbor_heuristic()
        self.nnh_objective_value = nnh_objective
        self.nnh_metrics = getattr(self, '_last_nnh_metrics', None)
        base_value = max(nnh_objective, 1e-6)
        self.init_pheromone_val = 1 / (base_value * self.node_num)
        self.pheromone_mat = np.ones((self.node_num, self.node_num)) * self.init_pheromone_val
        self.heuristic_info_mat = 1 / self.node_dist_mat

    # def _read_legacy_vehicle_info(self, file_path):
    #     """อ่าน vehicle_num และ vehicle_capacity จากไฟล์ instance ฟอร์แมตเดิม (บรรทัดที่ 5)"""
    #     with open(file_path, 'rt') as f:
    #         count = 1
    #         for line in f:
    #             if count == 5:
    #                 parts = line.split()
    #                 if len(parts) >= 2:
    #                     vehicle_num, vehicle_capacity = parts[:2]
    #                     return int(vehicle_num), int(vehicle_capacity)
    #                 break
    #             count += 1
    #     # fallback กันพัง
    #     return 9999, 10**9

    def copy(self, init_pheromone_val):
        new_graph = copy.deepcopy(self)

        # 信息素
        new_graph.init_pheromone_val = init_pheromone_val
        new_graph.pheromone_mat = np.ones((new_graph.node_num, new_graph.node_num)) * init_pheromone_val

        return new_graph

    def create_from_file(self, file_path):
        node_list = []
        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count >= 10:
                    parts = line.split()
                    if len(parts) >= 7:
                        node_list.append(parts)
                count += 1
        node_num = len(node_list)
        nodes = list(Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]),
                          float(item[4]), float(item[5]), float(item[6])) for item in node_list)

        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            node_dist_mat[i][i] = 1e-8
            for j in range(i+1, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]
        return node_num, nodes, node_dist_mat

    def _create_vehicle_types_from_file(self, file_path) -> List[Dict]:
        """
        รูปแบบไฟล์:
        type_id num capacity fixed_cost opera_cost

        ตัวอย่าง:
        0 5 100 500 1.0
        1 3 200 800 1.2
        2 2 300 1200 1.5
        """
        vtypes = []
        with open(file_path, 'rt') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or line.lower().startswith('type_id'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Vehicle line malformed: {line}")
                type_id = int(parts[0])
                num = int(parts[1])
                capacity = float(parts[2])
                fixed_cost = float(parts[3])
                var_cost = float(parts[4])
                vtypes.append({
                    "type_id": type_id,
                    "num": num,
                    "capacity": capacity,
                    "fixed_cost": fixed_cost,
                    "var_cost": var_cost
                })
        if not vtypes:
            raise ValueError("No vehicle types loaded.")
        return vtypes




    @staticmethod
    def _expand_vehicle_fleet(vehicle_types):
        fleet = []
        for vtype in vehicle_types:
            num = int(vtype.get("num", 0))
            base_info = {
                "type_id": vtype.get("type_id"),
                "capacity": float(vtype.get("capacity", 0)),
                "fixed_cost": float(vtype.get("fixed_cost", 0)),
                "var_cost": float(vtype.get("var_cost", 0))
            }
            fleet.extend([base_info.copy() for _ in range(num)])
        return fleet

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def local_update_pheromone(self, start_ind, end_ind):
        self.pheromone_mat[start_ind][end_ind] = (1-self.rho) * self.pheromone_mat[start_ind][end_ind] + \
                                                  self.rho * self.init_pheromone_val

    def global_update_pheromone(self, best_path, best_path_quality):
        """
        Update pheromone on the globally best path.
        """
        self.pheromone_mat = (1 - self.rho) * self.pheromone_mat

        quality = best_path_quality
        if quality is None or quality <= 0:
            quality = 1e-6

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho / quality
            current_ind = next_ind

    def nearest_neighbor_heuristic(self, max_vehicle_num=None):
        index_to_visit = list(range(1, self.node_num))
        current_index = 0
        current_load = 0
        current_time = 0
        travel_distance = 0.0
        travel_path = [0]

        total_travel_time = 0.0
        total_fixed_cost = 0.0
        total_operational_cost = 0.0

        fleet = list(getattr(self, 'vehicle_fleet', []))
        if fleet:
            vehicle_capacities = [v.get('capacity', self.vehicle_capacity) for v in fleet]
        else:
            vehicle_capacities = [self.vehicle_capacity]

        if max_vehicle_num is not None:
            vehicle_capacities = vehicle_capacities[:max_vehicle_num]
            if fleet:
                fleet = fleet[:max_vehicle_num]

        if not vehicle_capacities:
            raise ValueError('Nearest neighbour heuristic requires at least one vehicle.')

        vehicle_limit = max_vehicle_num if max_vehicle_num is not None else self.node_num
        vehicle_limit = max(1, vehicle_limit)

        current_vehicle_idx = 0
        current_vehicle_capacity = vehicle_capacities[0]
        current_vehicle_info = fleet[0] if fleet else {'fixed_cost': 0.0, 'var_cost': 0.0}
        routes_used = 0
        route_active = False

        while index_to_visit and routes_used < vehicle_limit:
            nearest_next_index = self._cal_nearest_next_index(
                index_to_visit,
                current_index,
                current_load,
                current_time,
                current_vehicle_capacity
            )

            if nearest_next_index is None:
                dist_to_depot = self.node_dist_mat[current_index][0]
                travel_distance += dist_to_depot
                total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist_to_depot
                total_travel_time += dist_to_depot

                travel_path.append(0)
                current_index = 0
                current_load = 0
                current_time = 0
                route_active = False

                routes_used += 1
                if not index_to_visit or routes_used >= vehicle_limit:
                    break

                if current_vehicle_idx + 1 < len(vehicle_capacities):
                    current_vehicle_idx += 1
                else:
                    current_vehicle_idx = len(vehicle_capacities) - 1
                    if not getattr(self, '_fleet_warning_issued', False):
                        print('Warning: available vehicle pool exhausted, reusing last capacity for additional routes.')
                        self._fleet_warning_issued = True

                current_vehicle_capacity = vehicle_capacities[current_vehicle_idx]
                if fleet:
                    current_vehicle_info = fleet[current_vehicle_idx]
                else:
                    current_vehicle_info = {'fixed_cost': 0.0, 'var_cost': 0.0}
            else:
                if current_index == 0 and not route_active:
                    total_fixed_cost += float(current_vehicle_info.get('fixed_cost', 0.0))
                    route_active = True

                demand = self.nodes[nearest_next_index].demand
                current_load += demand

                dist = self.node_dist_mat[current_index][nearest_next_index]
                wait_time = max(self.nodes[nearest_next_index].ready_time - current_time - dist, 0)
                service_time = self.nodes[nearest_next_index].service_time
                travel_wait = dist + wait_time

                current_time += travel_wait + service_time
                index_to_visit.remove(nearest_next_index)

                travel_distance += dist
                total_travel_time += travel_wait
                total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist
                travel_path.append(nearest_next_index)
                current_index = nearest_next_index

        dist_to_depot = self.node_dist_mat[current_index][0]
        travel_distance += dist_to_depot
        total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist_to_depot
        total_travel_time += dist_to_depot
        travel_path.append(0)

        if index_to_visit:
            raise ValueError('Nearest neighbour heuristic could not serve all customers with the available vehicle fleet.')

        vehicle_num = travel_path.count(0) - 1
        total_cost = total_travel_time + total_fixed_cost + total_operational_cost
        self._last_nnh_metrics = {
            'travel_time': total_travel_time,
            'travel_distance': travel_distance,
            'fixed_cost': total_fixed_cost,
            'operational_cost': total_operational_cost,
        }
        return travel_path, total_cost, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load, current_time, vehicle_capacity):
        """
        ????????next_index
        :param index_to_visit:
        :return:
        """
        nearest_ind = None
        nearest_distance = None

        for next_index in index_to_visit:
            demand = self.nodes[next_index].demand
            if vehicle_capacity is not None and current_load + demand > vehicle_capacity:
                continue

            dist = self.node_dist_mat[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time
            # ???????????,???????
            if current_time + dist + wait_time + service_time + self.node_dist_mat[next_index][0] > self.nodes[0].due_time:
                continue

            # ?????due time?????
            if current_time + dist > self.nodes[next_index].due_time:
                continue

            if nearest_distance is None or self.node_dist_mat[current_index][next_index] < nearest_distance:
                nearest_distance = self.node_dist_mat[current_index][next_index]
                nearest_ind = next_index

        return nearest_ind

class PathMessage:
    def __init__(self, path, objective, **metrics):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.objective = copy.deepcopy(objective)
            self.distance = self.objective
            self.used_vehicle_num = self.path.count(0) - 1
            self.metrics = copy.deepcopy(metrics)
        else:
            self.path = None
            self.objective = None
            self.distance = None
            self.used_vehicle_num = None
            self.metrics = {}

    def get_path_info(self):
        return self.path, self.objective, self.used_vehicle_num

    def get_metrics(self):
        return copy.deepcopy(self.metrics)

    def copy(self, init_pheromone_val):
        new_graph = copy.deepcopy(self)

        # 信息素
        new_graph.init_pheromone_val = init_pheromone_val
        new_graph.pheromone_mat = np.ones((new_graph.node_num, new_graph.node_num)) * init_pheromone_val

        return new_graph

    def create_from_file(self, file_path):
        node_list = []
        with open(file_path, 'rt') as f:
            count = 1
            for line in f:
                if count >= 10:
                    parts = line.split()
                    if len(parts) >= 7:
                        node_list.append(parts)
                count += 1
        node_num = len(node_list)
        nodes = list(Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]),
                          float(item[4]), float(item[5]), float(item[6])) for item in node_list)

        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            node_dist_mat[i][i] = 1e-8
            for j in range(i+1, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]
        return node_num, nodes, node_dist_mat

    def _create_vehicle_types_from_file(self, file_path) -> List[Dict]:
        """
        รูปแบบไฟล์ (CSV/TSV ก็ได้ แค่คั่นช่องว่าง) แนะนำให้มี header:
        type_id num capacity fixed_cost var_cost

        ตัวอย่าง:
        0 5 100 500 1.0
        1 3 200 800 1.2
        2 2 300 1200 1.5
        """
        vtypes = []
        with open(file_path, 'rt') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or line.lower().startswith('type_id'):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Vehicle line malformed: {line}")
                type_id = int(parts[0])
                num = int(parts[1])
                capacity = float(parts[2])
                fixed_cost = float(parts[3])
                var_cost = float(parts[4])
                vtypes.append({
                    "type_id": type_id,
                    "num": num,
                    "capacity": capacity,
                    "fixed_cost": fixed_cost,
                    "var_cost": var_cost
                })
        if not vtypes:
            raise ValueError("No vehicle types loaded.")
        return vtypes




    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def local_update_pheromone(self, start_ind, end_ind):
        self.pheromone_mat[start_ind][end_ind] = (1-self.rho) * self.pheromone_mat[start_ind][end_ind] + \
                                                  self.rho * self.init_pheromone_val

    def global_update_pheromone(self, best_path, best_path_quality):
        """
        Update pheromone on the globally best path.
        """
        self.pheromone_mat = (1 - self.rho) * self.pheromone_mat

        quality = best_path_quality
        if quality is None or quality <= 0:
            quality = 1e-6

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho / quality
            current_ind = next_ind

    def nearest_neighbor_heuristic(self, max_vehicle_num=None):
        index_to_visit = list(range(1, self.node_num))
        current_index = 0
        current_load = 0
        current_time = 0
        travel_distance = 0.0
        travel_path = [0]

        total_travel_time = 0.0
        total_fixed_cost = 0.0
        total_operational_cost = 0.0

        fleet = list(getattr(self, 'vehicle_fleet', []))
        if fleet:
            vehicle_capacities = [v.get('capacity', self.vehicle_capacity) for v in fleet]
        else:
            vehicle_capacities = [self.vehicle_capacity]

        if max_vehicle_num is not None:
            vehicle_capacities = vehicle_capacities[:max_vehicle_num]
            if fleet:
                fleet = fleet[:max_vehicle_num]

        if not vehicle_capacities:
            raise ValueError('Nearest neighbour heuristic requires at least one vehicle.')

        vehicle_limit = max_vehicle_num if max_vehicle_num is not None else self.node_num
        vehicle_limit = max(1, vehicle_limit)

        current_vehicle_idx = 0
        current_vehicle_capacity = vehicle_capacities[0]
        current_vehicle_info = fleet[0] if fleet else {'fixed_cost': 0.0, 'var_cost': 0.0}
        routes_used = 0
        route_active = False

        while index_to_visit and routes_used < vehicle_limit:
            nearest_next_index = self._cal_nearest_next_index(
                index_to_visit,
                current_index,
                current_load,
                current_time,
                current_vehicle_capacity
            )

            if nearest_next_index is None:
                dist_to_depot = self.node_dist_mat[current_index][0]
                travel_distance += dist_to_depot
                total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist_to_depot
                total_travel_time += dist_to_depot

                travel_path.append(0)
                current_index = 0
                current_load = 0
                current_time = 0
                route_active = False

                routes_used += 1
                if not index_to_visit or routes_used >= vehicle_limit:
                    break

                if current_vehicle_idx + 1 < len(vehicle_capacities):
                    current_vehicle_idx += 1
                else:
                    current_vehicle_idx = len(vehicle_capacities) - 1
                    if not getattr(self, '_fleet_warning_issued', False):
                        print('Warning: available vehicle pool exhausted, reusing last capacity for additional routes.')
                        self._fleet_warning_issued = True

                current_vehicle_capacity = vehicle_capacities[current_vehicle_idx]
                if fleet:
                    current_vehicle_info = fleet[current_vehicle_idx]
                else:
                    current_vehicle_info = {'fixed_cost': 0.0, 'var_cost': 0.0}
            else:
                if current_index == 0 and not route_active:
                    total_fixed_cost += float(current_vehicle_info.get('fixed_cost', 0.0))
                    route_active = True

                demand = self.nodes[nearest_next_index].demand
                current_load += demand

                dist = self.node_dist_mat[current_index][nearest_next_index]
                wait_time = max(self.nodes[nearest_next_index].ready_time - current_time - dist, 0)
                service_time = self.nodes[nearest_next_index].service_time
                travel_wait = dist + wait_time

                current_time += travel_wait + service_time
                index_to_visit.remove(nearest_next_index)

                travel_distance += dist
                total_travel_time += travel_wait
                total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist
                travel_path.append(nearest_next_index)
                current_index = nearest_next_index

        dist_to_depot = self.node_dist_mat[current_index][0]
        travel_distance += dist_to_depot
        total_operational_cost += float(current_vehicle_info.get('var_cost', 0.0)) * dist_to_depot
        total_travel_time += dist_to_depot
        travel_path.append(0)

        if index_to_visit:
            raise ValueError('Nearest neighbour heuristic could not serve all customers with the available vehicle fleet.')

        vehicle_num = travel_path.count(0) - 1
        total_cost = total_travel_time + total_fixed_cost + total_operational_cost
        self._last_nnh_metrics = {
            'travel_time': total_travel_time,
            'travel_distance': travel_distance,
            'fixed_cost': total_fixed_cost,
            'operational_cost': total_operational_cost,
        }
        return travel_path, total_cost, vehicle_num

    def _cal_nearest_next_index(self, index_to_visit, current_index, current_load, current_time, vehicle_capacity):
        """
        ????????next_index
        :param index_to_visit:
        :return:
        """
        nearest_ind = None
        nearest_distance = None

        for next_index in index_to_visit:
            demand = self.nodes[next_index].demand
            if vehicle_capacity is not None and current_load + demand > vehicle_capacity:
                continue

            dist = self.node_dist_mat[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time
            # ???????????,???????
            if current_time + dist + wait_time + service_time + self.node_dist_mat[next_index][0] > self.nodes[0].due_time:
                continue

            # ?????due time?????
            if current_time + dist > self.nodes[next_index].due_time:
                continue

            if nearest_distance is None or self.node_dist_mat[current_index][next_index] < nearest_distance:
                nearest_distance = self.node_dist_mat[current_index][next_index]
                nearest_ind = next_index

        return nearest_ind

class PathMessage:
    def __init__(self, path, objective, **metrics):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.objective = copy.deepcopy(objective)
            self.distance = self.objective
            self.used_vehicle_num = self.path.count(0) - 1
            self.metrics = copy.deepcopy(metrics)
        else:
            self.path = None
            self.objective = None
            self.distance = None
            self.used_vehicle_num = None
            self.metrics = {}

    def get_path_info(self):
        return self.path, self.objective, self.used_vehicle_num

    def get_metrics(self):
        return copy.deepcopy(self.metrics)

