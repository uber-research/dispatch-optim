import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_quant(a, q):
    """ Gives quantile of order 1 - q of probability distribution a.
    Args:
        a : np.ndarray, represents a probability distribution
        q : float, 1 - desired quantile

    Returns:
        float, quantile of order 1 - q

    """
    for t in range(len(a)):
        if np.sum(a[t:]) <= q:
            return t - 1

def normal_pdf(mu, sigma, bins):
    """ Approximates a normal distribution N(mu, sigma) over bins.
    Args:
        mu: float, mean of the distribution
        sigma: float, standard deviation of the distribution
        bins: np.ndarray, support, typically np.linspace() - like object

    Returns:
        y: PDF of a N(mu, sigma) over bins.
    """
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
    return y

def mixture(mu1, mu2, mu3, sig1, sig2, sig3, bins, w1=0.33, w2=0.33, w3=0.34):
    """ Approximate a mixture of 3 gaussian distributions over bins
    Args:
        mu1: float, mean of first gaussian
        mu2: float, mean of second gaussian
        mu3: float, mean of third gaussian
        sig1: float, standard deviation of first gaussian
        sig2: float, standard deviation of second gaussian
        sig3: float, standard deviation of third gaussian
        bins: np.ndarray, support, typically np.linspace() - like object
        w1, w2, w3: float, weights such that w1 + w2 + w3 == 1

    Returns:
        y: PDF of a mixture of Gaussian over bins.
           i.e. w1 * N(mu1, sig1) + w2 * N(mu2, sig2) + w3 * N(mu3, sig3)
    """
    y = w1 * normal_pdf(mu1, sig1, bins) + w2 * normal_pdf(mu2, sig2, bins) + w3 * normal_pdf(mu3, sig3, bins)
    return y

def new_demand(i, distri_new_demands, p_type, T):
    """Generate an individual demands for aerial transport.
       This is intended to be used to generate a single demand on a given leg.
    Args:
        i: int, ID of demand
        distri_new_demands: np.ndarray, distribution of new demands over time. Gives the likelihood
                            of new demands appearing over time. Here it will directly simulate the
                            ETA (mean of arrival times) at origin skyport.
        p_type: float, between 0 and 1. Gives the probability of being of type 1, i.e. hard constraint on arrival time
                at destination.
        T: list, time steps of service period.

    Returns:
        new_mu: float, mean arrival time at origin skyport
        new_std: float, standard deviation of arrival times at origin skyport
        new_lbda: float, parameters quantifying aversion to waiting.
        new_type: int, type of demand
        npax[0]: int, number of passengers in this demand
        i: int, ID of the demand.

    """
    npax = np.random.choice([1, 2, 3, 4], size = 1, p = np.array([0.7, 0.15, 0.1, 0.05]))
    new_lbda = 0.1
    new_type = np.random.choice([0, 1], p=[1-p_type, p_type])
    new_mu = np.random.choice(T, p=distri_new_demands)
    new_std = np.random.choice([7, 11, 8])
    return new_mu, new_std, new_lbda, new_type, npax[0], i

def generate_ind_demands(n, distri_new_demands, p_type, T, deltas):
    """ Generate a set of n individual demands.
        This is intended to be used to generate n demands on a given leg.
    Args:
        n: int, number of demands to generate
        distri_new_demands: np.ndarray, distribution of new demands over time. Gives the likelihood
                            of new demands appearing over time. Here it will directly simulate the
                            ETA (mean of arrival times) at origin skyport.
        p_type: float, between 0 and 1. Gives the probability of being of type 1, i.e. hard constraint on arrival time
                at destination.
        T: list, time steps of service period.
        deltas: dict, gives the maximum allowed probability of being late associated to each type of demand.

    Returns:
        points: list, contains all individuals demands generated as returned by new_demand()
        pt: list, contains all demands IDs
        costs: dict, contains the cost of pooling any two demand together : this is the time the earlier demand will have
               to wait because of being pooled with a later one.
        weights: dict, contains the number of pax for each demand ID
        quant: dict, contains the relevant quantile for each generated demand.
    """
    points = []
    pt = []
    for i in range(n):
        new_mu, new_std, new_lbda, new_type, npax, j = new_demand(i, distri_new_demands, p_type, T)
        points.append((new_mu, new_std, new_lbda, new_type, npax, j))
        pt.append(i)
    quantiles = []
    for p in points:
        a = normal_pdf(p[0], p[1], np.linspace(0, len(T), len(T)))
        quantiles.append(get_quant(a, deltas[p[3]]))
    weights = {}
    costs = {}
    quant = {}
    for i in range(len(points)):
        weights[i] = points[i][-2]
        for j in range(len(points)):
            costs[(i, j)] = abs(quantiles[i] - quantiles[j])
        quant[i] = quantiles[i]
    return points, pt, costs, weights, quant


def get_groups(match, points):
    """ Gives the groups (i.e. flights) created.
        Indicating all flights explicitly and the demands assgined to
        each flight.
    Args:
        match: obj, pooling milp (object in src/optim_pooling_complete.py)
        points: list, contains all demands

    Returns:
        G: dict, contains for all flights the list of assigned demand IDs
        F: dict, contains for all flights, the departure time.
    """
    G = {k: [] for k in match.groups}
    F = {k:0 for k in match.groups}
    for i in match.points:
        for k in match.groups:
            if match.x[i][k].varValue == 1:
                G[k].append(points[i])
                if match.quant[i] >= F[k]:
                    F[k] = match.quant[i]
    return G, F

def plot_groups(groups, flights, T, points, l):
    """ Function to plot a graph representing individual demands and colors
        to present groups created as well a flights.
    Args:
        groups: dict, contains for all flights the list of assigned demand IDs
        flights: dict, contains for all flights, the departure time.
        T: list, time steps of the service period.
        points: list, contains all individual demands.
        l: str, name of the leg

    Returns:
        None

    """
    colors=cm.rainbow(np.linspace(0,1, len(groups)))
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.title("Pooled ETA on leg " + str(l) + " | " + str(len(points)) + " random requests.")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("Time")

    for i in groups:
        for d in groups[i]:
            ax.plot(np.linspace(0, max(T), max(T)), normal_pdf(d[0], np.sqrt(d[1]), np.linspace(0, max(T), max(T))), label = str(i), color = colors[i], zorder = -10)

        ax.add_patch( Rectangle((flights[i] - 5, 0.),
                            5, 0.0,
                            fc ='none',
                            ec = colors[i],
                            fill = True,
                            lw = 8))

    plt.xticks([i for i in range(len(T))][::15], ['{:02d}:{:02d}'.format(*divmod(870 + i, 60)) for i in range(len(T))][::15])
    plt.ylabel("ETA likelihood")

    #plt.legend()
    plt.show()


def compute_exp_waiting(groups, flights, points, T):
    """ Compute the expected waiting time for each group created.
    Args:
        groups: dict, contains for all flights the list of assigned demand IDs
        flights: dict, contains for all flights, the departure time.
        points: list, contains all individual demands
        T: list, contains all time steps of the service period.

    Returns:
        W: dict, contain expected waiting time for each group.

    """
    W = {}
    T = np.linspace(0, len(T), len(T))
    for g in groups:
        for d in groups[g]:
            #compute expected waiting time
            dens = normal_pdf(d[0], np.sqrt(d[1]), T)
            if np.sum(dens) != 0:
              dens = dens / np.sum(dens)
            start = flights[g]
            expw = np.dot(np.flip(T[:start - 5]), dens[:start - 5])
            W[d] = expw
    return W
