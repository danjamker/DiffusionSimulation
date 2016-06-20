from __future__ import division

import json
import operator

import networkx as nx
import numpy as np
import scipy.stats
from collections import Counter
from collections import defaultdict

from networkx_additional_algorithms import structural_holes
from networkx_additional_algorithms import brokerage

class metric(object):

    roleTypes = {
        "coordinator": lambda pred, broker, succ: pred == broker == succ,
        "gatekeeper" 	 : lambda pred , broker ,succ: pred != broker == succ,
        "representative"	: lambda pred , broker , succ: pred == broker != succ,
        "consultant"		:lambda pred , broker, succ: pred == succ != broker,
        "liaison"		: lambda pred , broker ,succ: pred != succ and pred != broker and broker != succ,
    }

    def __init__(self, G, runDiamiter = True, group_name = "group", time_format = "%Y-%m-%d %H:%M:%S", edge_time_name = "createdat" ):

        self.group_name = group_name
        self.time_format = time_format
        self.edge_time_name = edge_time_name

        self.G = G

        self.run_d = runDiamiter
        if self.run_d is True:
            self.ud = self.G.to_undirected()

        self.Communities = list(set([attrdict[group_name] for n, attrdict in self.G.node.items()]))
        self.numberActivatedUsers = 0
        self.numberOfActivations = 0
        self.usagedominance = 0
        self.activationsPerCommunity = dict((k, 0) for k in self.Communities)
        self.activatedUsersPerCommunity = dict((k, 0) for k in self.Communities)
        self.userUsageDominance = 0
        self.usageEntorpy = 0
        self.userUsageEntorpy = 0
        self.ActivateionExposure = 0
        self.UserExposure = 0
        self.avrageActivateionExposure = 0
        self.avrageUserExposure = 0
        self.sequence = []
        self.sequence_community = []
        self.sequence_time = []
        self.ActivateionExposureArray = []
        self.UserExposureArray = []
        self.inffectedCommunities = 0
        self.numberActivatedUsersnorm = 0
        self.numberOfNodes = nx.number_of_nodes(G)

        #Time
        self.time_dif_sequence = []
        self.current_time = None
        self.early_spread_time = 0

        #Diamiter
        self.diamiter = 0

        #Surface
        self.surface_step = []
        self.surface = 0
        self.surface_set = set()

        #Wiener
        self.wiener_index_list = []
        self.cascade_edges = 0
        self.cascade_nodes = 0

        # User metrics
        self.pagerank = 0
        self.degrees = 0

        self.roles = { }
        for k in self.roleTypes.iterkeys():
            self.roles[k] = 0

        #Tag
        self.tag = []

    def add(self, n, step_time=None, tag=None):
        if n is not None:
            if self.G.has_node(n):

                node = self.G.node[n]
                self.sequence.append(n)
                self.sequence_community.append(node[self.group_name])
                self.sequence_time.append(step_time)

                if "pagerank" in node:
                    self.pagerank = node["pagerank"]
                self.degrees = self.G.degree(n)

                if node["activated"] == 1:
                    self.numberActivatedUsers += 1
                    self.numberActivatedUsersnorm = self.numberActivatedUsers / self.numberOfNodes
                    self.activatedUsersPerCommunity[node[self.group_name]] += 1
                    self.activatedUsersPerCommunity[node[self.group_name]] += 1

                self.numberOfActivations += 1
                self.activationsPerCommunity[node[self.group_name]] += 1

                #Compute time diffrence if provided
                if step_time != None:
                    self.sequence_time.append(step_time)

                    if self.current_time == None:
                        self.current_time = step_time
                    else:
                        self.time_dif_sequence.append(step_time - self.current_time)
                        self.current_time = step_time
                        self.early_spread_time = self.sequence_time[-1] - self.sequence_time[0]

                norm_activationsPerCommunity = [np.float64(self.activationsPerCommunity[c]) / self.numberOfActivations for c in self.Communities]
                self.usagedominance = max(norm_activationsPerCommunity)
                norm_activatedUsersPerCommunity = [np.float64(self.activatedUsersPerCommunity[c]) / self.numberActivatedUsers for c in self.Communities]
                self.userUsageDominance = max(norm_activatedUsersPerCommunity)

                # Shouldnt this be a list of number of the number of times a users has used an activation
                self.usageEntorpy = scipy.stats.entropy(norm_activationsPerCommunity)
                self.userUsageEntorpy = scipy.stats.entropy(norm_activatedUsersPerCommunity)

                #Compute Activations
                exposures = [self.G.node[ns]['activated'] for ns in self.G.neighbors(n) if
                             self.G.node[ns]['activated'] > 0]
                self.ActivateionExposure = np.sum(exposures)
                self.ActivateionExposureArray.append(self.ActivateionExposure)

                #compute user exposure
                self.UserExposure = len(exposures)

                self.inffectedCommunities = len({k: v for k, v in self.activatedUsersPerCommunity.items() if v > 0})
                self.inffectedCommunitiesnor = self.inffectedCommunities / len(self.activatedUsersPerCommunity)

                #Compute values for steps
                self.surface_set |= set([x for x in self.G.neighbors(n) if self.G.node[x]['activated'] == 0])
                if n in self.surface_set:
                    self.surface_set.remove(n)
                self.surface = len(self.surface_set)
                self.surface_step.append(self.surface)


                #Wiener indexx avrage
                sg = self.cascade_extrator(self.G, self.sequence, edge_time_format="%Y-%m-%d")
                for cc in list(nx.connected_component_subgraphs(sg.to_undirected())):
                    self.wiener_index_list.append(self.wiener_index(cc.to_undirected()))

                self.cascade_edges = nx.number_of_edges(sg)
                self.cascade_nodes = nx.number_of_nodes(sg)

                self.roles = self.extract_roles(sg)

                if self.run_d is True:
                    dist = [0]
                    for n1 in set(self.sequence):
                        if n1 != n:
                            try:
                                dist.append(nx.shortest_path_length(self.ud, source=n, target=n1))
                            except nx.NetworkXNoPath:
                                pass
                    if np.max(dist) > self.diamiter:
                        self.diamiter = np.max(dist)

    def cascade_extrator(self, G, sequence, node_time_attribute="time", edge_time_attribute = "createdat", edge_time_format = "%Y-%m-%d %H:%M:%S"):
        from datetime import datetime
        sg = G.subgraph(set(sequence)).copy()

        for s, t, d in sg.edges(data=True):
            if ( sg.node[s][node_time_attribute] > sg.node[t][node_time_attribute] ) or sg.node[s][node_time_attribute] < datetime.strptime(d[edge_time_attribute],edge_time_format):
                sg.remove_edge(*(s,t))

        return sg

    def asMap(self):
        r = {"number_activated_users": self.numberActivatedUsers,
                "number_activations": self.numberOfActivations,
                "usage_dominace": self.usagedominance,
                "user_usage_dominance": self.userUsageDominance,
                "activation_entorpy": self.usageEntorpy,
                "user_usage_entorpy": self.userUsageEntorpy,
                "activateion_exposure": self.ActivateionExposure,
                "user_exposure": self.UserExposure,
                "inffected_communities": self.inffectedCommunities,
                "surface": self.surface,
                "inffected_communities_normalised": self.inffectedCommunitiesnor,
                "node": self.sequence[-1],
                "community": self.sequence_community[-1],
                "time": self.sequence_time[-1],
                "number_activated_users_normalised": self.numberActivatedUsersnorm,
                "early_spread_time": self.early_spread_time,
                "pagerank": self.pagerank,
                "number_of_trees": len(self.wiener_index_list),
                "wiener_index_avrage": np.mean(self.wiener_index_list),
                "wiener_index_std": np.std(self.wiener_index_list),
                "degree": self.degrees,
                "cascade_nodes": self.cascade_nodes,
                "cascade_edges": self.cascade_edges,
                "diamiter": self.diamiter}

        for k, v in self.roles.iteritems():
            r[k] = v

        return r


    def wiener_index(self, G, weight=None):
        from itertools import chain

        is_directed = G.is_directed()
        if (is_directed and not nx.components.is_strongly_connected(G)) or \
                (not is_directed and not nx.components.is_connected(G)):
            return float('inf')
        pp = nx.shortest_paths.shortest_path_length(G, weight=weight)
        cc = [p.values() for k, p in pp.iteritems()]
        total = sum(chain.from_iterable(cc))
        # Need to account for double counting pairs of nodes in undirected graphs.
        return total if is_directed else total / 2

    def to_JSON(self):
        return json.dumps(self.__dict__)


    def extract_roles(self, sg):
        roles = dict((n, dict((role, 0) for role in self.roleTypes)) for n in sg)
        for node in sg:

            for successor in sg.successors(node):
                for predecessor in sg.predecessors(node):
                    if successor == predecessor or successor == node or predecessor == node: continue
                    if not (sg.has_edge(predecessor, successor)):
                        # found a broker!
                        # now which kind depends on who is in which group
                        roles[node][brokerage._RoleClassifier.classify(sg.node[predecessor][self.group_name],
                                                                       sg.node[node][self.group_name],
                                                                       sg.node[successor][self.group_name])] += 1
        return reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.iteritems()), roles.values())

class broker_metrics(metric):


    def __init__(self, G, attribute):
        super(self.__class__, self).__init__(G)
        self.attribute = attribute
        self.results = {}

    def add(self, no):
        self.sequence.append(no)
        sg = self.cascade_extrator()

        roles = self.extract_roles(sg)


        self.results = roles


        return roles

    def asMap(self):
        return self.results

