import random
import time


def read_file(rf_file_loc):
    """
    Reads data file

    :param rf_file_loc: String - Data file path
    :return rf_n: Int - Number of Locations
    :return rf_d_matrix: List - Distance Matrix
    :return rf_f_matrix: List - Flow Matrix
    """
    data = []
    with open(rf_file_loc, "r") as dat_file:
        for line in dat_file:  # Reads each line of file
            int_line = list(map(int, line.split()))  # Converts each number
            # in each line into an integer in a list
            data.append(int_line)
        dat_file.close()
    rf_n = data[0][0]  # Separates number of locations from data list
    rf_d_matrix = []
    rf_f_matrix = []
    for i in range(rf_n):
        rf_d_matrix.append(data[i+2])  # Forms distance matrix from data list
        rf_f_matrix.append(data[i+rf_n+3])  # Forms flow matrix from data list
    return rf_n, rf_d_matrix, rf_f_matrix


def init_p(ip_n):
    """
    Produces Initial Pheromones Randomly

    :param ip_n: Int - Number of Locations
    :return ip_r_matrix: List - Pheromone Matrix
    """
    ip_p_matrix = [[round(random.uniform(0, 1), 4) if i != j else 0 for j in
                    range(ip_n)] for i in range(ip_n)]
    return ip_p_matrix


def trans_probs_calc(tp_current, tp_n, tp_p_matrix):
    """
    Calculates transition probabilities from a given current node with given
    values.

    :param tp_current: Int - Index of current node
    :param tp_n: Int - Number of Locations
    :param tp_p_matrix: List - Pheromone Matrix
    :return probs: List - List of probabilities for each node
    """
    numerators = []
    for i in range(tp_n):
        numerators.append(tp_p_matrix[tp_current][i])  # Selects relevant
        # pheromone levels on paths going from current node
    denominator = sum(numerators)  # Sum of all relevant pheromone levels
    if denominator > 0:
        probs = list(map(lambda num: num/denominator, numerators))  #
        # Calculates probabilities for each path
    else:
        probs = numerators
    return probs


def next_node_calc(nn_n, nn_trans_probs):
    """
    Calculates the next node to visit with given transmission probabilities

    :param nn_n: Int - Number of Locations
    :param nn_trans_probs: List - Transmission Probabilities
    :return nn_index: Int - Index of next node to be visited
    """
    cumulative_probs = [0]
    for i in range(nn_n-1):
        cumulative_probs.append(cumulative_probs[i]+nn_trans_probs[i])  #
        # Calculates cumulative probabilities
    cumulative_probs.append(1)
    rand_num = random.uniform(0, 1)  # Generate random float between 0 and 1
    nn_index = 0
    while cumulative_probs[nn_index] <= rand_num:  # Find index of next path
        nn_index += 1
    nn_index -= 1
    return nn_index


def pheromone_update(pu_p_matrix, pu_n, pu_e, pu_paths, pu_path_costs):
    """
    Updates Pheromone matrix with given evaporation rate and ant deposits

    :param pu_p_matrix: List - Pheromone Matrix
    :param pu_n: Int - Number of Locations
    :param pu_e: Float - Evaporation Rate
    :param pu_paths: List - Ant Paths
    :param pu_path_costs: List - Cost of each ant path
    :return pu_p_matrix: List - Updated Pheromone Matrix
    """
    pu_p_matrix = [[round(1 - pu_e, 15) * pu_p_matrix[i][j] for j in range(
        pu_n)] for i in range(pu_n)]  # Evaporate all pheromones by given rate
    for path_i in range(len(pu_paths)):
        delta = 1 / pu_path_costs[path_i]  # Calculate deposit amount
        pu_path = pu_paths[path_i]
        for i in range(len(pu_paths[path_i])-1):
            pu_p_matrix[pu_path[i]][pu_path[i+1]] += delta  # Deposit on all
            # used path links by given ant
    return pu_p_matrix


def path_cost_calc(pc_paths, pc_d_matrix, pc_f_matrix):
    """
    Calculates the cost of each ant path given

    :param pc_paths: List - Ant Paths
    :param pc_d_matrix: List - Distance Matrix
    :param pc_f_matrix: List - Flow Matrix
    :return pc_costs: List - Ant path costs
    """
    pc_costs = []
    for pc_path in pc_paths:
        pc_cost = 0
        for i in range(len(pc_path) - 1):
            pc_distance = pc_d_matrix[pc_path[i]][pc_path[i + 1]]  # Finds
            # distance of link
            pc_flow = pc_f_matrix[i][i+1]  # Finds flow of link
            pc_cost += pc_distance*pc_flow  # Sums cost of each path
        pc_costs.append(pc_cost)
    return pc_costs


def rem_node(rn_node, rn_p_matrix):
    """
    Removes a given node from pheromone matrix

    :param rn_node: Int - Node index to be removed
    :param rn_p_matrix: List - Pheromone Matrix to be updated
    :return rn_matrix: List - Edited Matrix
    """
    for rn_line in rn_p_matrix:
        rn_line[rn_node] = 0  # Sets pheromones linking to given node as 0
    return rn_p_matrix


def run_aco(r_n, r_d_matrix, r_f_matrix, r_m, r_e, r_fitness_eval_num,
            r_trials_num):
    """
    Main function to run the Ant Colony Optimisation

    :param r_n: Int - Number of Locations
    :param r_d_matrix: List - Distance Matrix
    :param r_f_matrix: List - Flow Matrix
    :param r_m: Int - Number of Ants
    :param r_e: Float - Evaporation Rate
    :param r_fitness_eval_num: Int - Number of Evaluations
    :param r_trials_num: Int - Number of Trials
    :return cost_results: List - Best costs per trial
    :return path_results: List - Paths of best costs
    """
    cost_results = []
    path_results = []
    for trial in range(r_trials_num):
        print(trial)
        p_matrix = init_p(r_n)  # Produce random pheromone matrix
        path_costs = []
        ant_paths = []
        for eval_num in range(r_fitness_eval_num):  # Run each evaluation
            ant_paths = []
            for ant in range(r_m):  # Generate ant paths
                ant_path = []
                ant_p_matrix = []
                for row in p_matrix:
                    ant_p_matrix.append(list(row))  # Temporary copy of
                    # pheromone matrix
                current_node = random.randint(0, r_n-1)  # First node selection
                rem_node(current_node, ant_p_matrix)  # Remove first node
                # from available nodes
                ant_path.append(current_node)  # Add first node to ant's path
                while len(ant_path) < r_n:  # Create a path to all nodes
                    trans_probs = trans_probs_calc(current_node, r_n,
                                                   ant_p_matrix)  # Calculate
                    # transition probabilities
                    current_node = next_node_calc(r_n, trans_probs)  #
                    # Calculate next node to visit
                    rem_node(current_node, ant_p_matrix)  # Remove new node
                    ant_path.append(current_node)  # Add new node to ant's path
                ant_paths.append(ant_path)
            path_costs = path_cost_calc(ant_paths, r_d_matrix, r_f_matrix)
            # Calculate costs of each path
            p_matrix = pheromone_update(p_matrix, r_n, r_e, ant_paths,
                                        path_costs)  # Update the pheromone
            # matrix
        cost_results.append(min(path_costs))
        path_results.append(ant_paths[path_costs.index(min(path_costs))])
    return cost_results, path_results


def write_results(wr_file_loc, wr_data):
    """
    Write the result data to a given file

    :param wr_file_loc: String - Results File Path
    :param wr_data: List - List of Result Data
    """
    with open(wr_file_loc, "a") as results_file:  # Opens result file to
        # append
        results_file.write("\n"+wr_data[0]+"\n")
        results_file.write(wr_data[1]+"\n")
        results_file.write(wr_data[2]+"\n")
        results_file.write(wr_data[3] + "\n")
        results_file.close()


if __name__ == '__main__':
    file_loc = "Uni50a.dat"  # Data file path
    fitness_eval_num = 100  # Number of fitness evaluations
    trials_num = 5  # Number of trials per experiment
    n, d_matrix, f_matrix = read_file(file_loc)
    experiments_num = 4  # Number of experiments
    e = [0.9, 0.5, 0.9, 0.5]  # Evaporation Rates
    m = [100, 100, 10, 10]  # Numbers of Ants
    for exp in range(experiments_num):
        st = time.time()  # Takes start time for each experiment
        costs, paths = run_aco(n, d_matrix, f_matrix, m[exp], e[exp],
                               fitness_eval_num, trials_num)  # Runs ACO for
        # each experiment
        experiment_time = time.time()-st  # Calculates elapsed time for each
        # experiment
        results = [str(m[exp]), str(e[exp]),
                   time.strftime("%H:%M:%S", time.gmtime(experiment_time)),
                   str(costs)]  # Format Results
        write_results("results.txt", results)  # Write Results to file
