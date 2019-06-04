import math
import itertools
import csv
import numpy as np
import matplotlib.pyplot as plt
import collections


def average(xs):
    return sum(xs)/len(xs)


class HierarchicalClustering:
    def __init__(self, file_name, year):
        """Read data file, initialize the clustering"""
        f = open(file_name, "rt", encoding="latin1")

        self.year = str(year)

        self.tmp = list(csv.reader(f))

        self.header = [x.rstrip() for x in self.tmp[0][16:] if x != ""]
        self.tmppoints = [x[16:len(self.header)+16] for x in self.tmp[1:]] #if x[0] == self.year]

        self.points = []

        #replace missing values with -1
        for x in self.tmppoints:
            subtmp = []
            for y in x:
                if y == '':
                    subtmp.append(-1)
                else:
                    subtmp.append(int(y))
            self.points.append(subtmp)

        self.points = np.array(self.points)

        self.data = {}

        for x in range(len(self.header)):
            if not all(y == -1 for y in self.points[:,x]):
                self.data[self.header[x]] = self.points[:,x]

        #first each country represents a cluster
        self.clusters = [[name] for name in self.data.keys()]

        #dictionary of whole clustering, for graphical dendrogram
        self.history = {repr([x]): [None, 0, 0, 0, 0, 0, 0] for x in self.header}

        self.start_index = 1
        self.country_mapper = {}

        self.pref_countries = {}

        self.zero_countries = {}

        self.coordinates = []


    def row_distance(self, r1, r2):
        #eucledian distance between two vectors r1, r2 (two countries)
        dist = [(a - b) ** 2 if (a != -1 and b != -1) else -1 for a, b in zip(self.data[r1], self.data[r2])]

        ignored = dist.count(-1)

        if ignored != len(dist):
            return (math.sqrt(sum(x for x in dist if x != -1)/(len(dist)-ignored)))

    def cluster_distance(self, c1, c2):
        #distance between two clusters using average distance between all the combinations

        tmp = [c1, c2]

        ds = [self.row_distance(r1, r2)
                        for r1, r2 in itertools.product(*tmp)]

        if not all(x == None for x in ds):
            return average([x for x in ds if x != None])

    def closest_clusters(self):
        dis, pair = min((self.cluster_distance(c1, c2), [c1, c2])
                        for c1, c2 in itertools.combinations(self.clusters, 2)
                        if self.cluster_distance(c1, c2) != None)
        return dis, pair

    def run(self):
        while len(hc.clusters) > 1:
            dist, pair = self.closest_clusters()

            #set the distance where to stop the clustering
            #if dist > 3.79:
            #    break

            self.clusters.remove(pair[0])
            self.clusters.remove(pair[1])

            tmp = list(itertools.chain(pair[0], pair[1]))

            left = self.history[repr(pair[0])]
            right = self.history[repr(pair[1])]
            d_left = left[1]
            d_right = right[1]

            self.history[repr(tmp)] = [pair, dist, d_left, d_right, 0, 0, 0]

            self.clusters.append(tmp)

        #self.preferenced_countries()


    def preferenced_countries(self):
        for x in self.clusters:
            self.pref_countries[repr(x)] = []
            self.zero_countries[repr(x)] = []

            for y in x:
                self.pref_countries[repr(x)] += [self.tmp[i+1][1] for i,z in enumerate(self.data[y]) if z > 8]
                self.zero_countries[repr(x)] += [self.tmp[i + 1][1] for i, z in enumerate(self.data[y]) if z == 0]
            counted_pref = collections.Counter(self.pref_countries[repr(x)])
            counted_zero = collections.Counter(self.zero_countries[repr(x)])

            print(x, counted_pref.most_common(3), counted_zero.most_common(3))

    def text_dendrogram(self, name, indent):
        s = ""
        for x in range(indent):
            s += "    "
        s += "----"
        s += name
        print(s)

    def dendrogram_recursive(self, x, depth):
        y = self.history[repr(x)][0]

        if y == None:
            self.history[repr(x)][4] = self.start_index
            self.country_mapper[self.start_index] = x[0]
            self.start_index += 1
            self.text_dendrogram(repr(x), depth)
        else:
            self.dendrogram_recursive(y[0], depth+1)

            s = ""
            for t in range(depth):
                s += "    "
            s += "----|"
            print(s)

            self.dendrogram_recursive(y[1], depth+1)

            pos_left = self.history[repr(y[0])][4]
            pos_right = self.history[repr(y[1])][4]
            self.history[repr(x)][4] = (pos_left + pos_right)/2
            self.history[repr(x)][5] = pos_left
            self.history[repr(x)][6] = pos_right


    def graphical_dendrogram(self, x, dist):
        sons, distance, d_left, d_right, position, pos_left, pos_right = self.history[repr(x)]
        if sons == None:
            x1 = d_left
            x2 = dist
            y = position

            self.coordinates.append([(x1,x2),(y,y)])

        else:
            self.graphical_dendrogram(sons[0], distance)

            #horizonal left
            if d_left != 0:
                x1 = d_left
                x2 = distance
                y = pos_left
                self.coordinates.append([(x1, x2), (y, y)])

            #horizontal right
            if d_right != 0:
                x1 = d_right
                x2 = distance
                y = pos_right
                self.coordinates.append([(x1, x2), (y, y)])

            #vertical
            x1 = distance
            y1 = pos_left
            y2 = pos_right
            self.coordinates.append([(x1, x1), (y1, y2)])

            self.graphical_dendrogram(sons[1], distance)

    def draw_dendrogram(self):
        plt.figure()
        plt.yticks(range(len(self.country_mapper)+1))
        axes = plt.gca()
        axes.set_xlim([0, 5])
        axes.set_ylim([len(self.country_mapper)+1, 0])

        axes.set_title(self.year)

        for X,(y1, y2) in self.coordinates:
            axes.plot(X, (y1, y2), 'b')

        countries = [self.country_mapper[x] if x > 0 else '' for x in range(len(self.country_mapper)+1)]
        axes.set_yticklabels(countries)

        #for each year individually
        #plt.show(block=self.year == '2009')

        plt.show()


#each year individually
#for x in range(1998, 2010):

hc = HierarchicalClustering("data\eurovision-final.csv", '')
hc.run()
hc.dendrogram_recursive(hc.clusters[0], 0)
hc.graphical_dendrogram(hc.clusters[0], 0)
hc.draw_dendrogram()

#"data\eurovision-final-nowinners.csv"