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

    def __init__(self, G, runDiamiter = True, group_name = "group", time_format = "%Y-%m-%d %H:%M:%S", edge_time_name = "created_at", reciprical=False):

        self.group_name = group_name
        self.time_format = time_format
        self.edge_time_name = edge_time_name
        self.recip = reciprical
        self.G = G

        self.run_d = runDiamiter
        self.ud = self.G.to_undirected(reciprocal=self.recip)

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
        self.sequence_Induced_Degree = []
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
        self.step_dist = []


        #Surface
        self.surface_step = []
        self.surface = 0
        self.surface_set = set()

        #Wiener
        self.wiener_index_list = []
        self.cascade_edges = 0
        self.cascade_nodes = 0
        self.sg = []
        self.sg_numnodes = []
        self.fm = None
        self.dg = 0
        self.forest_density = 0
        self.broadcast_count = 0
        self.LargestTreeProp = 0

        # User metrics
        self.pagerank = 0
        self.degrees = 0

        self.roles = { }
        for k in self.roleTypes.iterkeys():
            self.roles[k] = 0

        #Tag
        self.tag = None

    def add(self, n, step_time=None, tag=None):
        if n is not None:
            if self.G.has_node(n):

                node = self.G.node[n]
                self.sequence.append(n)
                self.sequence_community.append(node[self.group_name])
                self.sequence_time.append(step_time)
                self.sequence_Induced_Degree.append(len(set(self.sequence).union(self.G.neighbors(n))))
                if "pagerank" in node:
                    self.pagerank = node["pagerank"]
                self.degrees = self.G.degree(n)
                self.tag = tag

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
                # "%Y-%m-%d"
                self.sg = self.cascade_extrator(self.G, self.sequence, edge_time_format=self.time_format, edge_time_attribute=self.edge_time_name)
                sg_ud = self.sg.to_undirected(reciprocal=self.recip)
                sg_con = list(nx.connected_component_subgraphs(sg_ud))
                self.sg_numnodes = [nx.number_of_nodes(xz) for xz in sg_con]
                self.wiener_index_list = [self.wiener_index(cc) for cc in sg_con]
                self.fm = self.forest_metrics(sg_con, self.ud)
                self.broadcast_count = len([z for z in self.depth_distribution(self.sg) if z == 1])

                self.cascade_edges = nx.number_of_edges(self.sg)
                self.cascade_nodes = nx.number_of_nodes(self.sg)
                self.vpi_list = [self.vpi(x,self.G) for x in sg_con]
                self.deg1 = [x for x in list(self.sg.degree().values()) if x == 1]
                self.roles = self.extract_roles(self.sg)
                self.forest_density = np.divide(nx.number_of_edges(self.sg),(nx.number_of_nodes(self.sg) * (nx.number_of_nodes(self.sg) -1)))
                self.LargestTreeProp = np.divide(max(self.sg_numnodes), self.cascade_nodes)

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

                if len(self.sequence) == 1:
                    self.step_dist.append(0)
                else:
                    self.step_dist.append(nx.shortest_path_length(self.ud, source=self.sequence[-1], target=self.sequence[-2]))

    def depth_distribution(self, sg, node_time_attribute="time"):
        depth = []
        for cc in nx.connected_component_subgraphs(sg.to_undirected(reciprocal=self.recip)):
            n, t = cc.nodes()[0], sg.node[cc.nodes()[0]][node_time_attribute]
            for nt in cc.nodes()[1:]:
                if sg.node[nt][node_time_attribute] < t:
                    n = nt
                    t = sg.node[nt][node_time_attribute]

            depth.append(max(nx.shortest_path_length(cc, n).values()))
        return depth

    def cascade_extrator(self, G, sequence, node_time_attribute="time", edge_time_attribute = "created_at", edge_time_format = "%Y-%m-%d %H:%M:%S"):
        from datetime import datetime
        sg = G.subgraph(set(sequence)).copy()

        for s, t, d in sg.edges(data=True):
            if ( sg.node[s][node_time_attribute] > sg.node[t][node_time_attribute] ) or sg.node[s][node_time_attribute] < datetime.strptime(d[edge_time_attribute],edge_time_format):
                sg.remove_edge(*(s,t))

        # for n in sg.nodes():
        #     if len(sg.in_edges(n)) > 0:
        #         tmp = sg.in_edges(n)
        #         so, to = sg.in_edges(n)[0]
        #         dif = sg.node[n][node_time_attribute] - sg.node[so][node_time_attribute]
        #         for s, t in sg.in_edges(n)[1:]:
        #             if sg.node[n][node_time_attribute] - sg.node[s][node_time_attribute] < dif:
        #                 sg.remove_edge(*(so, to))
        #                 dif = sg.node[n][node_time_attribute] - sg.node[s][node_time_attribute]
        #                 so = s
        #             else:
        #                 sg.remove_edge(*(s, t))


        return sg

    def forest_metrics(self, cascades, G):
        def numeric_compare(x, y):
            return nx.number_of_nodes(x) - nx.number_of_nodes(y)

        ordered_cascades = sorted(cascades, cmp=numeric_compare)

        larger_cascade = ordered_cascades[0]
        path_length = []
        path_list = []
        nodes = set(nx.nodes(ordered_cascades[0]))

        for c in ordered_cascades[1:]:
            tmp_path = None
            for n in nx.nodes(larger_cascade):
                for n2 in nx.nodes(c):
                    tmp = nx.shortest_path(G, n, n2)
                    if tmp_path is not None and len(tmp) < len(tmp_path):
                        tmp_path = tmp
                    elif tmp_path is None:
                        tmp_path = tmp

            path_length.append(len(tmp_path))
            path_list.append(tmp_path)
            nodes |= set(tmp_path)
            larger_cascade = G.subgraph(nodes).copy()
            nodes = set(nx.nodes(larger_cascade))

        connector_sg = nx.subgraph(G, set([item for sublist in path_list for item in sublist]))
        connector_edge_count = nx.number_of_edges(connector_sg)
        connector_node_count = nx.number_of_nodes(connector_sg)

        node_ratio = np.divide(sum([nx.number_of_nodes(s) for s in ordered_cascades]) , nx.number_of_nodes(larger_cascade))
        edge_ratio = np.divide(sum([nx.number_of_edges(s) for s in ordered_cascades]) , nx.number_of_edges(larger_cascade))

        degree2 = len([z for z in list(larger_cascade.degree().values()) if z > 2])

        dencity = np.divide(nx.number_of_edges(larger_cascade),(nx.number_of_nodes(larger_cascade) * (nx.number_of_nodes(larger_cascade) -1)))
        mean_path_length = np.mean([len(x) for x in path_list])
        number_of_triangels = nx.triangles(larger_cascade)

        return nx.number_of_edges(larger_cascade), connector_node_count, connector_edge_count, node_ratio, edge_ratio, degree2, sum(number_of_triangels.values()), mean_path_length, dencity

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
                "diamiter": self.diamiter,
                "vpi": np.mean(self.vpi_list),
                "largets_tree": max(self.sg_numnodes),
                "iso_trees": len([x for x in self.sg_numnodes if x == 1]),
                "three_trees": len([x for x in self.sg_numnodes if x > 3]),
                "forest_uc_connector_path_node_count":self.fm[1],
                "forest_uc_connector_path_edge_count": self.fm[2],
                "forest_uc_node_ratio": self.fm[3],
                "forest_uc_degree2": self.fm[4],
                "forest_uc_triangels": self.fm[5],
                "forest_uc_mean_path_length": self.fm[6],
                "forest_uc_dencity": self.fm[7],
                "forest_deg1": self.deg1,
                "forest_dencity":self.forest_density,
                "forest_broadcast_count": self.broadcast_count,
                "forest_largestTreeProp": self.LargestTreeProp,
                "step_distance": self.step_dist[-1],
                "tag":self.tag
             }

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


    def vpi(self, cascade, G):

        v = []
        for n in cascade.nodes():
            active = len([x for x in G.neighbors(n) if G.node[x]['activated'] > 0])
            nabours = len(G.neighbors(n))

            r = np.divide(active, nabours)
            v.append(np.power(r, active))
        return np.log(np.product(v))*-1


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

