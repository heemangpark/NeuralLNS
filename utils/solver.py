import copy
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


def save_map(grid, filename, save_dir):
    f = open(save_dir + '{}.map'.format(filename), 'w')
    f.write('type four-directional\n')
    f.write('height {}\n'.format(grid.shape[0]))
    f.write('width {}\n'.format(grid.shape[1]))
    f.write('map\n')

    # creating map from grid
    map_dict = {0: '.', 1: '@'}
    for r in range(grid.shape[0]):
        line = grid[r]
        l = []
        for g in line:
            l.append(map_dict[g])
        f.write(''.join(l) + '\n')

    f.close()


def save_scenario(agent_pos, total_tasks, scenario_name, row, column, save_dir):
    f = open(save_dir + '{}.scen'.format(scenario_name), 'w')
    f.write('version 1\n')
    for a, t in zip(agent_pos, total_tasks):
        task = t[0]
        dist = abs(np.array(a) - np.array(t)).sum()  # Manhattan dist
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scenario_name), row, column, a[1],
                                                                  a[0], task[1], task[0], dist)
        f.write(line + "\n")
    f.close()


def one_to_one_scen(a_coords, t_coords, scen_name, row, column, save_dir):
    f = open(save_dir + '{}.scen'.format(scen_name), 'w')
    f.write('version 1\n')
    for a, t in zip(a_coords, t_coords):
        dist = abs(np.array(a) - np.array(t)).sum()
        line = '1 \t{} \t{} \t{} \t{} \t{} \t{} \t{} \t{}'.format('{}.map'.format(scen_name), row, column, a[1],
                                                                  a[0], t[1], t[0], dist)
        f.write(line + "\n")
    f.close()


def read_trajectory(path_file_dir):
    f = open(path_file_dir, 'r')
    lines = f.readlines()
    agent_traj = []

    for i, string in enumerate(lines):
        curr_agent_traj = []
        splitted_string = string.split('->')
        for itr, s in enumerate(splitted_string):
            if itr == len(splitted_string) - 1:
                continue
            if itr == 0:
                tup = s.split(' ')[-1]
            else:
                tup = s

            ag_loc = [int(i) for i in tup[1:-1].split(',')]
            curr_agent_traj.append(ag_loc)
        agent_traj.append(curr_agent_traj)

    f.close()

    return agent_traj


def to_solver(task_in_seq, assignment):
    s_in_tasks = [[] for _ in range(len(assignment))]
    for a, t in assignment.items():
        if len(t) == 0:
            pass
        else:
            __t = list()
            for _t in t:
                __t += task_in_seq[list(_t.keys())[0]]
            s_in_tasks[a] = __t
    return s_in_tasks


def solver(map, agents, tasks, save_dir, exp_name, ret_log=False):
    if not os.path.exists(save_dir + exp_name):
        os.makedirs(save_dir + exp_name)

    time_log = dict()
    s_agents = copy.deepcopy(agents)
    todo = copy.deepcopy(tasks)
    seq_paths = [[list(agents[a])] for a in range(len(agents))]
    total_cost, itr, T = 0, 0, 0

    s = time.time()
    while sum([len(t) for t in todo]) != 0:
        itr += 1
        s_tasks = list()
        for a, t in zip(s_agents, todo):
            if len(t) == 0:
                s_tasks.append([list(a)])
            else:
                s_tasks.append([t[0]])
        save_map(map, exp_name, save_dir + exp_name + '/')
        save_scenario(s_agents, s_tasks, exp_name, map.shape[0], map.shape[1], save_dir + exp_name + '/')

        c = [os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/pbs'),
             "-m",
             save_dir + exp_name + '/' + '{}.map'.format(exp_name),
             "-a",
             save_dir + exp_name + '/' + '{}.scen'.format(exp_name),
             "-o",
             save_dir + exp_name + '/' + '{}.csv'.format(exp_name),
             "--outputPaths",
             save_dir + exp_name + '/' + '{}_paths_{}.txt'.format(exp_name, itr),
             "-k", "{}".format(len(s_agents)),
             "-t", "{}".format(1)]

        process_out = subprocess.run(c, capture_output=True)
        text_byte = process_out.stdout.decode('utf-8')

        if (text_byte[37:44] != 'Succeed') & ret_log:
            if os.path.exists(save_dir + exp_name + '/'):
                shutil.rmtree(save_dir + exp_name + '/')
            return 'retry', 'retry', 'retry'
        elif text_byte[37:44] != 'Succeed':
            if os.path.exists(save_dir + exp_name + '/'):
                shutil.rmtree(save_dir + exp_name + '/')
            return 'retry', 'retry'

        traj = read_trajectory(save_dir + exp_name + '/{}_paths_{}.txt'.format(exp_name, itr))
        len_traj = [len(t) - 1 for t in traj]
        d_len_traj = [l for l in len_traj if l not in {0}]
        next_t = np.min(d_len_traj)
        T += next_t

        fin_id = list()
        for e, t in enumerate(traj):
            if len(t) == 1:
                fin_id.append(False)
            else:
                fin_id.append(t[next_t] == s_tasks[e][0])
        fin_ag = np.array(range(len(s_agents)))[fin_id]

        for a_id in range(len(s_agents)):
            if a_id in fin_ag:
                if len(todo[a_id]) == 0:
                    pass
                else:
                    ag_to = todo[a_id].pop(0)
                    time_log[tuple(ag_to)] = T
                    s_agents[a_id] = ag_to
            else:
                if len_traj[a_id] == 0:
                    pass
                else:
                    s_agents[a_id] = traj[a_id][next_t]

            seq_paths[a_id] += traj[a_id][1:next_t + 1]

        total_cost += next_t * len(d_len_traj)

        if (time.time() - s) > 100:
            break

    if os.path.exists(save_dir + exp_name + '/'):
        shutil.rmtree(save_dir + exp_name + '/')

    if ret_log:
        return total_cost, seq_paths, time_log
    else:
        return total_cost, seq_paths


def assignment_to_id(n_ag, assignment):
    keys, values = [], [[] for _ in range(n_ag)]
    for ag_idx, tasks in enumerate(assignment.values()):
        keys.append(ag_idx)
        for task in tasks:
            values[ag_idx].append(list(task.keys())[0])
    assign_id = dict(zip(keys, values))
    return assign_id


def id_to_assignment(assign_id, task_coords):
    id = copy.deepcopy(assign_id)
    for a in id.keys():
        for idx, t_id in enumerate(id[a]):
            id[a][idx] = {t_id: task_coords[t_id]}

    return id


def one_step_solver(map, agents, tasks, save_dir, exp_name):
    if not os.path.exists(save_dir + exp_name):
        os.makedirs(save_dir + exp_name)

    save_map(map, exp_name, save_dir + exp_name + '/')
    one_to_one_scen(agents, tasks, exp_name, map.shape[0], map.shape[1], save_dir + exp_name + '/')

    c = [os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/pbs'),
         "-m",
         save_dir + exp_name + '/' + '{}.map'.format(exp_name),
         "-a",
         save_dir + exp_name + '/' + '{}.scen'.format(exp_name),
         "-o",
         save_dir + exp_name + '/' + '{}.csv'.format(exp_name),
         "--outputPaths",
         save_dir + exp_name + '/' + '{}_paths.txt'.format(exp_name),
         "-k", "{}".format(len(agents)),
         "-t", "{}".format(1)]

    process_out = subprocess.run(c, capture_output=True)
    if os.path.exists(process_out.args[8]):
        f = open(process_out.args[8], 'rb')
        paths = str(f.read()).split('\\n')[:-1]
        costs = [p.count('->') - 1 for p in paths]
    else:
        costs = 'retry'

    if os.path.exists(save_dir + exp_name + '/'):
        shutil.rmtree(save_dir + exp_name + '/')

    return costs
