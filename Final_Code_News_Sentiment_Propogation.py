import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
negative_utilities = []
positive_utilities = []
in_degrees_by_level = {}
out_degrees_by_level = {}


class TraversalResult:
    def __init__(self, level, retweets, retweeted_nodes, unretweeted_nodes):
        self.level = level
        self.retweets = retweets
        self.retweeted_nodes = retweeted_nodes
        self.unretweeted_nodes = unretweeted_nodes


def return_neighbors(G, seednode):
    neighbors = nx.all_neighbors(G, seednode)
    nb = []
    for n in neighbors:
        nb.append(n)
    return nb


def return_neighbor_attribute(attribute):
    newatt = attribute
    # interaction matrix between different attributes
    interaction_matrix = [
        [0.426, 0.213, 0.319, 0.042],
        [0.213, 0.426, 0.042, 0.319],
        [0.319, 0.042, 0.426, 0.213],
        [0.042, 0.319, 0.213, 0.426]
    ]
    weights = interaction_matrix[newatt]
    possible_values = [0, 1, 2, 3]
    attribute = random.choices(possible_values, weights)
    return attribute


def max_indegree(G):
    max = 0
    for node in list(G.nodes()):
        in_degree = G.in_degree(node)
        if in_degree > max:
            max = in_degree
    return max


def construct_graph():
    G = nx.DiGraph()
    inputfile = "C:\\Users\\senth\\Downloads\\ohyeahbaby.csv"
    edges_source = pd.read_csv(inputfile, usecols=[0], sep=",")
    edges_destination = pd.read_csv(inputfile, usecols=[1], sep=",")
    temp_source_nodes = edges_source.values.tolist()
    temp_destination_nodes = edges_destination.values.tolist()
    for i in range(len(temp_source_nodes)):
        current_source_node = temp_source_nodes[i]
        current_destination_node = temp_destination_nodes[i]
        G.add_edge(current_source_node[0], current_destination_node[0])
    G.remove_nodes_from(nx.isolates(G))
    G = nx.convert_node_labels_to_integers(G)
    num_nodes = 235
    max_attribute_count = [39, 107, 24, 65]
    node_count_threshold = 200
    age = {}
    political_extremity = {}
    attribute_dictionary = {}
    seednode = random.randint(0, num_nodes-1)
    possible_values = [2, 0, 1, 3]
    # 0=YC
    # 1=YE
    # 2=OC
    # 3=OE

    weights = [10, 16.5, 27.8, 45.7]
    seedattribute = random.choices(possible_values, weights)
    attribute_dictionary[seednode] = seedattribute
    currentnode = seednode
    currentattribute = seedattribute
    numclassifiednodes = 0
    classified_nodes = []
    classified_nodes.append(seednode)
    attribute_counts = [0, 0, 0, 0]
    possible_values = [2, 0, 1, 3]
    while len(classified_nodes) < num_nodes:
        neighbors = return_neighbors(G, currentnode)
        for i in neighbors:
            for j in range(4):
                if attribute_counts[j] >= max_attribute_count[j] and j in possible_values:
                    possible_values.remove(j)
            if i not in classified_nodes:
                # TODO: Fix this hack
                if isinstance(currentattribute, list):
                    currentattribute = currentattribute[0]
                possible_attribute = return_neighbor_attribute(
                    currentattribute)
                possible_attribute = possible_attribute[0]
                while possible_attribute not in possible_values:
                    possible_attribute = return_neighbor_attribute(
                        currentattribute)
                    possible_attribute = possible_attribute[0]
                attribute_counts[possible_attribute] = attribute_counts[possible_attribute] + 1
                attribute_dictionary[i] = possible_attribute
                numclassifiednodes = numclassifiednodes + 1
                classified_nodes.append(i)
        currentnode = random.choices(neighbors)
        currentnode = currentnode[0]
        currentattribute = attribute_dictionary[currentnode]
        if len(possible_values) == 1:
            for i in range(num_nodes):
                if i not in classified_nodes:
                    attribute_dictionary[i] = possible_values[0]
                    classified_nodes.append(i)
                    numclassifiednodes = numclassifiednodes+1
        if len(classified_nodes) > node_count_threshold:
            attributes_remaining = []
            for k in range(4):
                attributes_remaining.append(
                    max_attribute_count[k] - attribute_counts[k])
            remaining_indexes = []
            for i in range(4):
                if attributes_remaining[i] > 0:
                    remaining_indexes.append(i)
            new_attributes_remaining = [
                value for value in attributes_remaining if value > 0]
            for k in range(num_nodes):
                if k not in classified_nodes:
                    att = random.choices(
                        remaining_indexes, new_attributes_remaining)
                    att = att[0]
                    index = remaining_indexes.index(att)
                    num = new_attributes_remaining[index]
                    if num > 1:
                        new_attributes_remaining[index] = num-1
                    else:
                        remaining_indexes.remove(att)
                    attribute_dictionary[k] = att
                    classified_nodes.append(k)
                    numclassifiednodes = numclassifiednodes+1

    for i in range(num_nodes):
        if attribute_dictionary[i] == 0:
            age[i] = 'Young'
            political_extremity[i] = 'Center'
        elif attribute_dictionary[i] == 1:
            age[i] = 'Young'
            political_extremity[i] = 'Extreme'
        elif attribute_dictionary[i] == 2:
            age[i] = 'Old'
            political_extremity[i] = 'Center'
        else:
            age[i] = 'Old'
            political_extremity[i] = 'Extreme'
    nx.set_node_attributes(G, age, 'Age')
    nx.set_node_attributes(G, political_extremity, 'Political Extremity')
    nx.set_node_attributes(G, attribute_dictionary, 'Number')
    return G


def get_characteristics(G, seednode):
    return G.nodes[seednode]


def return_base_probability(characteristics, positiveornegative):
    oldextremistpositive = 0.2
    oldextremistnegative = 0.65
    youngextremistpositive = 0.5
    youngextremistnegative = 1
    oldneutralpositive = 0.65
    oldneutralnegative = 0.56
    youngneutralpositive = 0.7
    youngneutralnegative = 0.61
    if characteristics['Age'] == 'Young':
        if characteristics['Political Extremity'] == 'Center':
            if positiveornegative == 'Positive':
                return youngneutralpositive
            else:
                return youngneutralnegative
        else:
            if positiveornegative == 'Positive':
                return youngextremistpositive
            else:
                return youngextremistnegative
    else:
        if characteristics['Political Extremity'] == 'Center':
            if positiveornegative == 'Positive':
                return oldneutralpositive
            else:
                return oldneutralnegative
        else:
            if positiveornegative == 'Positive':
                return oldextremistpositive
            else:
                return oldextremistnegative


def calculate_utility(indegree, retweet_percentage, base_probability, totaledges, num_nodes, num_retweeted, max_indegree):
    b = random.gauss(0, 0.1)
    b = b * indegree
    b = b / max_indegree
    cost = 0.695
    utility = base_probability + b + num_retweeted - cost
    return utility


def rewire_edge(G):
    current_node = random.randint(0, 234)
    neighbors = get_following(G, current_node)
    random_neighbor = random.choices(neighbors)
    random_neighbor = random_neighbor[0]
    if current_node == random_neighbor:
        rewire_edge(G)
    G.remove_edge(current_node, random_neighbor)
    if nx.is_strongly_connected(G) == True:
        random_neighbor = random.choices(neighbors)
        random_neighbor = random_neighbor[0]
        G.add_edge(current_node, random_neighbor)
        return G
    else:
        G.add_edge(current_node, random_neighbor)
        rewire_edge(G)


def traverse_graph_spread_news_helper(G, start_node, positive_or_negative):
    final_values = []
    maximum_indegree = max_indegree(G)
    #todo: remove
    retweets = 0
    next_level = 1
    nodes_to_traverse = {}
    retweeted_nodes = []
    un_retweeted_nodes = []
    followers = get_followers(G, start_node)
    # gets all nodes that follow source node
    nodes_to_traverse[next_level] = []
    in_degrees_by_level[next_level] = []
    out_degrees_by_level[next_level] = []

    nodes_to_traverse[next_level + 1] = []
    in_degrees_by_level[next_level + 1] = []
    out_degrees_by_level[next_level + 1] = []

    for i in followers:
        nodes_to_traverse[next_level].append(i)
    retweeted_nodes.append(start_node)

    while len(nodes_to_traverse[next_level]) > 0:
        current_node = nodes_to_traverse[next_level].pop(0)
        if current_node not in retweeted_nodes:
            retweet_percent = 'NaN'
            share_decision = get_share_decision(
                G, current_node, positive_or_negative, retweet_percent, retweeted_nodes, maximum_indegree)
            if share_decision == 1:
                # add followers to nodes_to_travers
                followers = get_followers(G, current_node)
                retweets = retweets + 1
                # gets all nodes that follow source node
                for i in followers:
                    if i not in nodes_to_traverse[next_level + 1]:
                        nodes_to_traverse[next_level + 1].append(i)
                retweeted_nodes.append(current_node)
            else:
                un_retweeted_nodes.append(current_node)
            # Get indegree and outdegree
            in_degrees_by_level[next_level].append(G.in_degree(current_node))
            out_degrees_by_level[next_level].append(G.out_degree(current_node))

            # print(G.out_degree(current_node), end = ", ")
        if len(nodes_to_traverse[next_level]) == 0:
            next_level = next_level + 1
            nodes_to_traverse[next_level + 1] = []
            in_degrees_by_level[next_level + 1] = []
            out_degrees_by_level[next_level + 1] = []

    final_values.append(next_level-1)
    final_values.append(retweets)
    return TraversalResult(next_level - 1, retweets, retweeted_nodes, un_retweeted_nodes)


def get_followers(G, current_node):
    followers_list = []
    connected_nodes = list(G.in_edges(current_node))
    for i in range(len(list(connected_nodes))):
        temp_source_nodes = list(connected_nodes[i])
        # gets all nodes that follow source node
        followers_list.append(temp_source_nodes[0])
    return followers_list


def get_following(G, current_node):
    following_list = []
    connected_nodes = list(G.out_edges(current_node))
    for i in range(len(list(connected_nodes))):
        temp_source_nodes = list(connected_nodes[i])
        # gets all nodes that follow source node
        following_list.append(temp_source_nodes[0])
    return following_list


def get_share_decision(G, current_node, positive_or_negative, retweet_percent, recieved_nodes, max_indegree):
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    in_degree = G.in_degree(current_node)
    num_retweeted = 0
    characteristics = get_characteristics(G, current_node)
    following = get_following(G, current_node)
    for i in following:
        if i in recieved_nodes:
            num_retweeted = num_retweeted+1
    if num_retweeted > 1:
        print(num_retweeted)
    if positive_or_negative == True:
        # base probability according to attributes and type of news
        base_probability = return_base_probability(characteristics, 'Positive')
    else:
        base_probability = return_base_probability(characteristics, 'Negative')
    utility = calculate_utility(
        in_degree, retweet_percent, base_probability, num_edges, num_nodes, num_retweeted, max_indegree)
    if positive_or_negative == True:
        positive_utilities.append(utility)
    else:
        negative_utilities.append(utility)
    distance_from_zero = abs(utility)
    # trembling hands: probability of node with utility < 1 sharing, or node with utility > 1 not sharing
    trembling_hands_probability = 1 / distance_from_zero
    trembling_hands_probability = trembling_hands_probability/3
    # convert probability into decimal
    trembling_hands_probability = trembling_hands_probability / 100
    normal_hands_probability = 1 - trembling_hands_probability
    share = [0, 1]
    if utility > 0:
        weights = [trembling_hands_probability, normal_hands_probability]
        final_decision = random.choices(share, weights)
        # return final_decision
    elif utility == 0:
        weights = [0.5, 0.5]
        final_decision = random.choices(share, weights)
        # return final_decision
    elif utility < 0:
        weights = [normal_hands_probability, trembling_hands_probability]
        final_decision = random.choices(share, weights)
        # return final_decision
    return final_decision[0]


def avg(n):
    sum = 0
    for i in n:
        sum = sum+i
    return sum/len(n)


G = construct_graph()
print(nx.diameter(G))
print(max_indegree(G))
num_nodes = G.number_of_nodes()
seed_node = random.randint(0, num_nodes - 1)
positive_retweets_array = []
positive_iterations_array = []
zeros_array = []
ones_array = []
twos_array = []
threes_array = []
degree_dict = {}
for k in range(200):
    traversal_results = traverse_graph_spread_news_helper(G, seed_node, False)
    level = traversal_results.level
    positive_retweets = traversal_results.retweets
    positive_retweets_array.append(positive_retweets)
    level = traversal_results.level
    positive_iterations_array.append(level)
    number_vals = []
    retweeted_nodes = traversal_results.retweeted_nodes
    for i in retweeted_nodes:
        characteristics = get_characteristics(G, i)
        num = characteristics['Number']
        number_vals.append(num)
    ones = 0
    twos = 0
    threes = 0
    zeros = 0
    for i in number_vals:
        if i == 1:
            ones = ones + 1
        elif i == 2:
            twos = twos + 1
        elif i == 3:
            threes = threes + 1
        else:
            zeros = zeros + 1
    zeros = zeros / 39
    ones = ones / 107
    twos = twos / 24
    threes = threes / 65
    ones_array.append(ones)
    zeros_array.append(zeros)
    twos_array.append(twos)
    threes_array.append(threes)
    #values = [zeros,ones,twos,threes]
    # plt.bar(names,values)
    # plt.show()
    values = []
    names = []
    print('Nodes: ', G.number_of_nodes())
    for level in in_degrees_by_level.keys():
        # print(i.out_degrees)
        out_degrees = out_degrees_by_level[level]
        sum = 0
        for j in out_degrees:
            sum = sum + j
        if len(out_degrees) == 0:
            break
        avg_out_degree = sum / len(out_degrees)
        values.append(avg_out_degree)
        names.append(str(level))
    degree_dict[k] = []
    degree_dict[k].extend(values)
names = ['Young + Center', 'Young + Extreme', 'Old + Center', 'Old + Extreme']
zeros = avg(zeros_array)
ones = avg(ones_array)
twos = avg(twos_array)
threes = avg(threes_array)
values = [zeros, ones, twos, threes]
print(values)
plt.bar(names, values)
plt.show()
names = []
values = []
print('avg retweets', avg(positive_retweets_array))
print('avg iterations', avg(positive_iterations_array))
reshaped_dict = {}
max = max(degree_dict.values(), key=len)
lena = len(max)
for i in range(lena):
    reshaped_dict[i] = []
for key in degree_dict.keys():
    current_list = degree_dict[key]
    current_list = list(current_list)
    for i in range(len(current_list)):
        reshaped_dict[i].append(current_list[i])
for key in reshaped_dict.keys():
    values.append(avg(reshaped_dict[key]))
    names.append(str(key))
print(values)
plt.bar(names, values)
plt.show()
