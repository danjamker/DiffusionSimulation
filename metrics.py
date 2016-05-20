from __future__ import division

import json
import operator

import networkx as nx
import numpy as np
import scipy.stats
class metric:
    def __init__(self, G):
        self.G = G
        self.Communities = list(set([attrdict['community'] for n, attrdict in self.G.node.items()]))
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
        self.window = 100
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

        # User metrics
        self.pagerank = 0
        self.degrees = 0

        #Tag
        self.tag = None

    def add(self, n, step_time=None, tag=None):
        if n is not None:
            if self.G.has_node(n):

                node = self.G.node[n]
                self.sequence.append(n)
                self.sequence_community.append(node["community"])
                self.sequence_time.append(step_time)
                if "pagerank" in node:
                    self.pagerank = node["pagerank"]
                self.degrees = self.G.degree(n)

                if node["activated"] == 1:
                    self.numberActivatedUsers += 1
                    self.numberActivatedUsersnorm = self.numberActivatedUsers / self.numberOfNodes
                    self.activatedUsersPerCommunity[node["community"]] += 1

                self.numberOfActivations += 1
                self.activationsPerCommunity[node["community"]] += 1

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
                self.tag.append(tag)


    def asMap(self):
        return {"numberActivatedUsers": self.numberActivatedUsers,
                "numberOfActivations": self.numberOfActivations,
                "usagedominance": self.usagedominance,
                "userusagedominance": self.userUsageDominance,
                "usageEntorpy": self.usageEntorpy,
                "userUsageEntorpy": self.userUsageEntorpy,
                "ActivateionExposure": self.ActivateionExposure,
                "UserExposure": self.UserExposure,
                "inffectedCommunities": self.inffectedCommunities,
                "surface": self.surface,
                "inffectedCommunitiesnor": self.inffectedCommunitiesnor,
                "node": self.sequence[-1],
                "community": self.sequence_community[-1],
                "time": self.sequence_time[-1],
                "numberActivatedUsersnorm": self.numberActivatedUsersnorm,
                "early_spread_time": self.early_spread_time,
                "pagerank": self.pagerank,
                "degree": self.degrees,
                "tag":self.tag}

    def to_JSON(self):
        return json.dumps(self.__dict__)
