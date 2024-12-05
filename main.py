import argparse
import json
from timeit import default_timer as timer
from inspect import getmembers, isfunction
import csv  # 추가된 모듈

from numpy import number

from ContainerMatrix import ContainerMatrix
from Model import Model
import Constraints as Constraints

def positive_int(x):
    i = int(x)
    if i < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive integer value" % x)
    return i
def positive_float(x):
    f = float(x)
    if f < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % x)
    return f

def print_cond(b=True, str='', **kwargs):
    if b:
        print(str, **kwargs)

def enforce_container_lifetime_restrictions(model : Model, matrix : ContainerMatrix, labels, index_lookup, initial_container_positions, shipments, time):
    for container in labels:
        # This is the life cycle of the containers
        dead       = [None, None]
        in_        = [None, None]
        alive      = [None, None]
        out        = [None, None]
        dead_again = [None, None]

        dead = [0, 0]

        for shipment in shipments:
            if alive == [None, None] and container in (c[0] for c in initial_container_positions):
                dead = [0, 0]
                in_  = [0, 0]
                alive = [0, shipment["duration"]]
                continue

            if in_ == [None, None]:
                if "in" not in shipment:
                    dead[1] += shipment["duration"]
                elif container not in shipment["in"]:
                    dead[1] += shipment["duration"]
                elif container in shipment["in"]:
                    in_ = [dead[1], dead[1] + shipment["duration"]]
                else:
                    raise Exception("Unreachable code")
                
            elif alive == [None, None]:
                alive = [in_[1], in_[1] + shipment["duration"]]

            elif out == [None, None]:
                if "out" not in shipment:
                    alive[1] += shipment["duration"]
                elif container not in shipment["out"]:
                    alive[1] += shipment["duration"]
                elif container in shipment["out"]:
                    out = [alive[1], alive[1] + shipment["duration"]]
                    dead_again = [alive[1] + shipment["duration"], time]
                    break
                else:
                    raise Exception("Unreachable code")
        
        if out == [None, None] and dead_again == [None, None]:
            out = [time, time]
            dead_again = [time, time]
        
        # print(f"{dead=} {in_=} {alive=} {out=} {dead_again=}")
        # input()

        for t in range(time):
            if dead[0] <= t < dead[1] or dead_again[0] <= t < dead_again[1]:
                model.Add(matrix.lifetime[t][index_lookup[container]] == 0)
            elif alive[0] <= t < alive[1]:
                model.Add(matrix.lifetime[t][index_lookup[container]] == 1)
        
    final_shipment = shipments[-1]
    if "in" in final_shipment and "out" in final_shipment:
        for c in final_shipment["in"]:
            model.Add(matrix.lifetime[-1][index_lookup[c]] == 1)
        for c in final_shipment["out"]:
            model.Add(matrix.lifetime[-1][index_lookup[c]] == 0)

def enforce_container_loading_restrictions(model : Model, matrix : ContainerMatrix, shipments):
    move_counter = 0
    for shipment in shipments:
        next_move_counter = move_counter+shipment["duration"]

        if "in" not in shipment:
            model.Add(sum(matrix.insert[move_counter:next_move_counter]) == 0)
            model.Add(sum(matrix.remove[move_counter:next_move_counter]) == 0)
        
        move_counter += next_move_counter

    in_ = 0
    out = 0
    for shipment in shipments:
        print(shipment)
        if "in" in shipment:
            in_ += len(shipment["in"])
        if "out" in shipment:
            out += len(shipment["out"])
    
    model.Add(sum(matrix.insert) == in_)
    model.Add(sum(matrix.remove) == out)

def enforce_weight_restrictions(model : Model, matrix : ContainerMatrix, weights, index_lookup : dict):
    weights = {label:1 for label in index_lookup} if weights == False else weights
    weight_array = [0] * len(index_lookup)

    for c, weight in weights.items():
        weight_array[index_lookup[c]] = weight
    
    for t in range(matrix.t):
        for s in range(matrix.s):
            for h in range(matrix.h):
                for c in range(matrix.c):
                    container_is_here = model.NewBoolVar('b')
                    model.AddIf(matrix.get(t, c, s, h) == 1, container_is_here)
                    model.AddIf(matrix.get(t, c, s, h) == 0, model.Not(container_is_here))

                    for container in range(matrix.c):
                        for height in range(matrix.h):
                            if container != c and height < h and weight_array[container] < weight_array[c]:
                                model.AddIf(matrix.get(t, container, s, height) == 0, container_is_here)

def minimize_ship_loading_time(model : Model, matrix : ContainerMatrix, shipments):
    ship_idles = 0

    move_counter = 0
    for shipment in shipments:
        next_move_counter = move_counter+shipment["duration"]

        if "in" in shipment:
            ship_idles += sum(matrix.idle[move_counter:next_move_counter])
        
        move_counter += next_move_counter

    model.Maximize(ship_idles * 100000 + sum(matrix.idle)) # By maximizing the number of idle actions, we minimize emplaces and removes and inserts

def load_from_json(args : object, logs : bool = False, visualize : bool = True) -> dict:
    with open(args.path) as f:
        data = json.load(f)
        
        print_cond(logs, "Input file loaded: '" + args.path + "'")

    length, height = data["dimensions"]
    shipments = data["shipments"]
    time = sum(shipment["duration"] for shipment in shipments)
    weights = data["weights"] if "weights" in data else False

    initial_container_positions = data["containers"]

    containers = [i[0] for i in initial_container_positions]
    for shipment in shipments:
        if "in" in shipment:
            containers += shipment["in"]

    index_lookup = {label : index for index, label in enumerate(containers)}
    labels = containers

    # Model creation phase
    constraint_start = timer()

    model = Model(args.solver)
    print_cond(logs, "Generating matrix")
    matrix = ContainerMatrix(model, time, len(containers), length, height)

    # Constraint application phase

    print_cond(logs, "Implementing matrix constraints")
    constraints = getmembers(Constraints, isfunction)
    for _, constraint in constraints: 
        print_cond(logs, _, end=" ", flush=True)
        constraint(model, matrix)
    print_cond(logs)

    print_cond(logs, "Setting initial container positions")
    for container, (label, stack, height) in enumerate(initial_container_positions):
        model.Add(matrix.get(0, container, stack, height) == 1)
    
    print_cond(logs, "Setting initial container lifetime")
    for i, container in enumerate(containers):
        if container in [i[0] for i in initial_container_positions]:
            model.Add(matrix.lifetime[0][i] == 1)
        else:
            model.Add(matrix.lifetime[0][i] == 0)

    print_cond(logs, "Enforcing container lifetime restrictions")
    enforce_container_lifetime_restrictions(model, matrix, labels, index_lookup, initial_container_positions, shipments, time)

    print_cond(logs, "Enforcing container movement restrictions")
    enforce_container_loading_restrictions(model, matrix, shipments)

    print_cond(logs, "Enforcing weight restrictions")
    enforce_weight_restrictions(model, matrix, weights, index_lookup)
    
    minimize_ship_loading_time(model, matrix, shipments)
    
    constraint_end = timer()
    # Solve and result phas

    solution = model.Solve(args.time, args.execfile)

    if solution['status'] == model.OPTIMAL or solution['status'] == model.FEASIBLE:
        if logs:
            print('Solution time (s):', solution['time'])
            print('Objective value:', solution['objective'], 'OPTIMAL' if solution['status'] == model.OPTIMAL else '')
            matrix.print_solution(model, labels=labels)


        # --- 컨테이너 이동 경로 출력 및 CSV 저장 ---
        log_file = "movement_log.csv"
        with open(log_file, mode="w", newline="") as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(["time", "container", "stack", "height", "status"]) 

            print("\nContainer Movement Log:")
            previous_positions = {}  # 이전 시간의 위치를 저장

            for t in range(matrix.t):  # 시간 단위 반복
                print(f"\nTime {t}: Container stacking status")  # 현재 시간 출력

                stacking_status = [["." for _ in range(matrix.s)] for _ in range(matrix.h)]  # 적재 상태 초기화

                for c in range(matrix.c):  # 각 컨테이너 반복
                    current_position = None  # 현재 위치

                    for s in range(matrix.s):  # 각 스택 반복
                        for h in range(matrix.h):  # 스택 내 높이 반복
                            if model.Value(matrix.get(t, c, s, h)) == 1:
                                current_position = (s, h)  # 현재 위치 저장

                                # 적재 상태 갱신
                                stacking_status[h][s] = labels[c]  # 높이 h, 스택 s에 컨테이너 표시

                                # CSV에 기록
                                log_writer.writerow([t, labels[c], s, h, "idle"])
                                print(f"At time {t}, Container {labels[c]} is at Stack {s}, Height {h}")

                    # 현재 위치를 이전 위치로 업데이트
                    if current_position:
                        previous_positions[c] = current_position

                # 적재 상태 출력
                for row in reversed(stacking_status):  # 높이가 위에서 아래로 출력되도록 역순 출력
                    print(" | ".join(row))
        print(f"\nMovement log saved to {log_file}")


        if visualize:
            matrix.visualize(model, shipments, labels=labels)

        print("Solver: ", args.solver, "Constraint Time: ", constraint_end-constraint_start, solution)
    else:
        print("No feasible solution found")
    
    return solution
    
    
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Run the solver for the Container Stacking Problem')

    my_parser.add_argument('-solver',
        metavar='--solver-package',
        type=str,
        required=True,
        choices=['ortools', 'cplex'],
        help="choice of solver to solve input problem. Currently supports 'ortools' and 'cplex'")

    my_parser.add_argument('-path',
        metavar='--input-path',
        type=str,
        default='inputs/input.json',
        help="the path to the file with the input problem (.json). By default is 'inputs/input.json'")

    my_parser.add_argument('-benchmark',
        metavar='--benchmark-runs',
        nargs='?',
        type=positive_int,
        default=0,
        help="number of runs for benchmarking time (recommended: 5)")
    
    my_parser.add_argument('-time',
        metavar='--max-time',
        nargs='?',
        type=positive_float,
        default=None,
        help="time limit (in seconds) to return solution. May return sub-optimal solution, or none at all")

    my_parser.add_argument('-execfile', 
        metavar='--cplex-execfile',
        nargs = '?',
        type=str,
        default=None,
        help="path for the CPLEX engine's executable. By default is '/opt/ibm/ILOG/CPLEX_Studio201/cpoptimizer/bin/x86-64_linux/cpoptimizer'"
        )
    args = my_parser.parse_args()

    if args.benchmark == 0:
        load_from_json(args, logs=True, visualize=True)

    else:
        import timeit
        print("Solver [", args.solver, "]")
        print("Number runs [", args.benchmark, "]")

        t = timeit.Timer(lambda: load_from_json(args, logs=False, visualize=False)['status'])
        print("Avg time (s)", t.timeit(number=args.benchmark)/args.benchmark)
        # (total_time, sol) = t.timeit(args.benchmark)
        # print("Time avg(s): ", total_time/args.benchmark)
