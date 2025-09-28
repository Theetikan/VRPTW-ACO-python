from vrptw_base import VrptwGraph
from basic_aco import BasicACO
import sys


def iter_routes(path, depot_id=0):
    route = []
    vehicle_idx = 0
    for node in path:
        if node == depot_id:
            if route:
                yield vehicle_idx, [depot_id, *route, depot_id]
                route = []
                vehicle_idx += 1
        else:
            route.append(node)
    if route:
        yield vehicle_idx, [depot_id, *route, depot_id]


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    vehicle_file = './vehicles.txt'
    ants_num = 150
    max_iter = 400
    beta = 1.7
    alpha = 1.2
    q0 = 0.85
    show_figure = True

    graph = VrptwGraph(file_path, vehicle_file)
    basic_aco = BasicACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, alpha=alpha, q0=q0,
                         whether_or_not_to_show_figure=False)

    basic_aco.run_basic_aco()

    if not basic_aco.best_path:
        print('No feasible route found.')
        sys.exit(0)

    print('Sub-routes:')
    fleet = graph.vehicle_fleet or [{'type_id': 'N/A', 'capacity': graph.vehicle_capacity}]
    for idx, route in iter_routes(basic_aco.best_path):
        vehicle_info = fleet[idx] if idx < len(fleet) else fleet[-1]
        type_id = vehicle_info.get('type_id', 'N/A')
        capacity = vehicle_info.get('capacity')
        if capacity is None:
            capacity_str = 'n/a'
        else:
            capacity_str = 'inf' if capacity == float('inf') else f"{capacity:g}"
        print(f"Route {idx + 1} (Vehicle type {type_id}, capacity {capacity_str}): {route}")
