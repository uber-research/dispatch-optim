
import numpy as np
import utils_pooling as up
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class K_MeansW():
    """ Implement a K-means clustering to be applied to Gaussian Probablity distributions
        using Wasserstein distance.
        This object is useful to the Recursive Splitting algorithm to pool individual
        demands.

    """
    def __init__(self, k=2, tol=0.001, max_iter=300):
        """ K means algo using Wasserstein distance for matching algo
            TODO : adapt to generic case for non gaussian distri.

            Args:
                    k: int, number of groups for the K-Means
                    tol: float, tolerance for stopping criterion
                    max_iter: int, maximum number of iterations before stopping.
                    Since the clustering problem is simple, and that we want to use it
                    with k = 2, this number can be low.

            Returns:
                    None
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def w2_dist(self, p1, p2):
        """ Compute L2-wasserstein distance between two gaussian distribution using L2 cost.

            Args:
                    p1: list or tuple of len 2, (mean, std) of first gaussian
                    p2: list or tuple of len 2, (mean, std) of second gaussian

            Returns:
                    float, L2-wasserstein distance between p1 and p2.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def compute_barycenter(self, mus, sigs):
        """ Given a two list of floats, this function returns the means of the two lists.
            Usefule for barycenters in the k-means.

            Args:
                    mus: list, contains means. Intended to contain means of point within a group
                    sigs: list, contains stds. Intended to contain stds of point within a group

            Returns:
                    mu_bar: float, mean of mus
                    sig: float, mean of sigs
        """
        mu_bar = np.mean(mus)
        sig = np.mean(sigs)
        return mu_bar, sig

    def fit(self, points):
        """ Perform K-means using L2-Wasserstein distance. With k=2, this will split the data points in 2 groups
            putting points which are closer together.

            Args:
                    points: list, contains all individual demands as output by utils_pooling.new_demand()

            Returns:
                    None

        """
        self.centroids = {}
        p = [i for i in range(self.k)]
        np.random.shuffle(p)
        for i in p:
            self.centroids[i] = (points[i][0], points[i][1])

        for it in range(self.max_iter):
            self.groups = {}

            for i in range(self.k):
                self.groups[i] = []

            for ms in points:
                g = list(self.centroids.keys())[0]
                dist = self.w2_dist(self.centroids[g], ms)
                for gp in self.centroids:
                    dp = self.w2_dist(self.centroids[gp], ms)
                    if dp < dist:
                        dist = dp
                        g = gp
                self.groups[g].append(ms)

            for g in self.groups:
                mus, sigs = [], []
                for p in self.groups[g]:
                    mus.append(p[0])
                    sigs.append(p[1])
                self.centroids[g] = self.compute_barycenter(mus, sigs)



class MATCHING_HEURI():
    """ This class implements the matching heuristic : Recursive Splitting.
        It will will recursively split the points in two groups until the matching condition
        are satisfied.
        Splitting is done using the W-Kmeans implemented above.


    """
    def __init__(self,
                capacity,
                delta,
                distri_new_demands,
                p_type,
                T,
                deltas,
                leg,
                aversion=lambda x, lbda : (np.exp(lbda * x) - np.exp(lbda*10)) * (x > 10)):
        """ Initialize the class params

            Args:
                 capacity: int, maximum number of pax in one helicopter.
                 delta: float, maximum spread of one group
                 distri_new_demands: list, gives the likelihood of new demand appearing over the service period
                 p_type: dict, gives probability of demands being of one of the types
                 T: list, time steps of the service periods
                 deltas: dict, gives the maximum accepted probability of being late for each type of demand
                 leg: str, leg on which the matching is performed.

            Returns:
                 None
        """

        self.capacity = capacity
        self.delta = delta
        self.p_type = p_type
        self.distri_new_demands = distri_new_demands / np.sum(distri_new_demands)
        self.T = T
        self.deltas = deltas
        self.l = leg
        self.aversion = aversion

    def new_demand(self, i):
        """ Generate a new demand

            TODO: Remove this as it is duplicated in utils_pooling.py
            Args:
                 i: int, ID of demand generated.

            Returns:


        """
        npax = np.random.choice([1, 2, 3, 4], size = 1, p = np.array([0.7, 0.15, 0.1, 0.05]))
        new_lbda = 0.1
        new_type = np.random.choice([0, 1], p=[1-self.p_type, self.p_type])
        new_mu = np.random.choice(self.T, p=self.distri_new_demands)
        new_std = np.random.choice([7, 10, 5])

        return new_mu, new_std, new_lbda, new_type, npax[0], i

    def generate_demands(self, n):
        """ Generate n new demands

            TODO: Remove this as it is duplicated in utils_pooling.py

            Args:
                  i: int, number of demands to be generated.

            Returns:

        """
        points = []
        for i in range(n):
            new_mu, new_std, new_lbda, new_type, npax, j = self.new_demand(i)
            a = up.normal_pdf(new_mu, new_std, np.linspace(0, len(self.T), len(self.T)))
            q = up.get_quant(a, self.deltas[new_type])
            points.append((new_mu, new_std, new_lbda, new_type, npax, j, q))
        return points


    def plot_points(self, points):
        """ Viz function
            Plots a set of demands, represented by their ETAs.

            Args:
                    points: list, contains all points to be plotted.
                         Each point in points is of the format as output by new_demand()

            Returns:
                    None

        """
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.title("ETA distribution for request on leg " + str(self.l) + " | " + str(len(points)) + " random requests.")
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("Time")

        for i in range(len(points)):
            ax.plot(np.linspace(0, len(self.T), len(self.T)), up.normal_pdf(points[i][0], np.sqrt(points[i][1]), np.linspace(0, len(self.T), len(self.T))))
            #ax.axvline(x=quantiles[i])
        plt.xticks([i for i in range(len(self.T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(self.T))][::30])
        plt.ylabel("ETA likelihood")
        #plt.legend()
        plt.show()
        return None


    def split(self, pop):
        """ Given a set a points, this function will split them in 2 using the 2-means.

            Args:
                pop: list, contains initial population of points.

            Returns:
                child1: list, contains first group created
                child2: list, contains second group created

        """
        if len(pop) == 2:
            if self.check_cap(pop, self.capacity) and self.check_spread(pop, self.delta):
                return [[]], pop
            else:
                return [pop[0]], [pop[1]]
        kw = K_MeansW(k = 2)
        kw.fit(pop)
        G = kw.groups
        child1, child2 = G[0], G[1]
        return child1, child2

    def check_cap(self, g, C):
        """ Given a group g, this function checks that the total number of pax in this group
            does not exceed the helicopter capacity C.

            Args:
                    g: list, contains all points present in g. Each element must be
                        of the format as output by new_demand().
                    C: int, capacity of helicopters. In the future can be heterogenous.

            Returns:
                    bool: True iif capacity constraint is satisfied.

        """
        if len(g) == 0:
            return True
        return sum(np.array(g)[:, -3]) <= C

    def check_spread(self, g, delta):
        """ Given a group g, this function checks that the spread within this group
            does not exceed the maximum allowed, delta.
            While the delta is constant, the spread is computed by getting the maximum waiting time in the group,
            and weigthing it by the aversion function which is parametrized by a demand-dependent parameter.

            Args:
                    g: list, contains all points present in g. Each element must be
                        of the format as output by new_demand().
                    delta: float, maximum spread of a group

            Returns:

                    bool: true iif spread constraint is satisfied.

        """
        if len(g) == 0:
            return True
        # points are like (new_mu, new_std, new_lbda, new_type, npax, j, q)
        max_q = np.max(np.array(g)[:, -1])
        for ele in g:
            time_diff = max_q - ele[-1]
            if self.aversion(time_diff, ele[2]) > delta:
                return False
        return True


    def tree(self, points, C, delta):
        """ This function is the actual recursive splitting. The initial population is recursively splitted into two groups,
            therefore building a tree, until all child satisfy all constraints.

            Args:
                    points: list, contains all

            Returns:

        """

        if len(points) <= 1:
            #terminal node
            return points

        child1, child2 = self.split(points)
        #if len(points) == 3:

        if len(child1) == 0 or len(child2) == 0:
            return child1, child2
        # if len(points) == 2:
        #     return child1, child2
        c1, s1 = self.check_cap(child1, C), self.check_spread(child1, delta)
        c2, s2 = self.check_cap(child2, C),  self.check_spread(child2, delta)

        if not(c1 * s1) and not(c2 * s2):
            return self.tree(child1, C, delta), self.tree(child2, C, delta)
        if not(c1 * s1) and c2 * s2:
            return self.tree(child1, C, delta), child2
        if not(c2 * s2) and c1 * s1:
            return self.tree(child2, C, delta), child1
        if c2 * s2 and c1 * s2:
            return child1, child2

    def flatten(self, nest):
        """ This function will flatten a nest of tuple. Typically the output of self.tree

            Args:
                    nest : tuple of tuple nested

            Returns:
                    Generator returning flattened element

        """
        for v in nest:
            if isinstance(v, tuple):
                yield from self.flatten(v)
            else:
                yield (v)

    def get_groups(self, nest):
        """
            This function computes flights and group assignements and should take as input
            the ouptput of self.tree

            Args:
                    nest: tuple of tuples, output of self.tree
            Returns:
                    G: dict, key are group index and values are points in the
                    F: dict, key are group index and value are flight departure time (lower bound on time windows)
        """
        G = {}
        F = {}
        gp = [k for k in self.flatten(nest) if k != []]
        # print(gp)
        for i in range(len(gp)):
            G[i] = gp[i]
            F[i] = np.max(np.array(gp[i])[:, -1])
        return G, F


    def pool(self, n):
        """ This function will randomly generate n individual demands and pool them using the RS algo.

            Args:
                    n: int, number of individual demands to simulate
            Returns:
                    G: dict, key are group index and values are points in the
                    F: dict, key are group index and value are flight departure time (lower bound on time windows)
                    points: list, contains all individual demands generated
        """
        if n == 1:
            points = self.generate_demands(n)
            G = {0: points}
            F = {0: points[0][-1]}
            return G, F, points

        #--
        points = self.generate_demands(n)
        nest = self.tree(points, self.capacity, self.delta)
        G, F = self.get_groups(nest)
        return G, F, points


    def plot_groups(self, groups, flights, points):
        """ This function plots the individual arrival time distributions (as gaussian).

            Args:
                    groups: dict, key are group index and values are points in the
                    fligths: dict, key are group index and value are flight departure time (lower bound on time windows)
                    points: list, contains all individual demands generated
            Returns:
                    fig: matplotlib figure object, display individuals demands as well as flights created.

        """
        #names = [str(n) for n in list(groups.keys())]
        colors=cm.rainbow(np.linspace(0,1, len(groups)))
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.title("Pooled ETA on leg " + str(self.l) + " | " + str(len(points)) + " random requests.")
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("Time")

        for i in groups:
            for d in groups[i]:
                ax.plot(np.linspace(0, len(self.T), len(self.T)), up.normal_pdf(d[0], np.sqrt(d[1]), np.linspace(0, len(self.T), len(self.T))), label = str(i), color = colors[i], zorder = -10)
        #for i in groups:
            ax.add_patch( Rectangle((flights[i] - 5, 0.),
                                5, 0.0,
                                fc ='none',
                                ec = colors[i],
                                fill = True,
                                lw = 8))

        plt.xticks([i for i in range(len(self.T))][::30], ['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(self.T))][::30])
        plt.ylabel("ETA likelihood")
        return fig
