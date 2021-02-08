import pulp as plp
import time

class MATCHING_ILP():
  """ This class contains the ILP implementation of the pooling problem.
      Defined in the tech report (see overleaf / pdf), it aims a minizing the number of
      flights created while allocating each demand to a flight and respecting waiting time constraints.
      This ILP is implemented for a fixed number of flights, it should be ran several times with several values
      for this variable (increasing the number of flights) and stopping whenever a solution is found.
  """
  def __init__(self, points, weights, capacity, groups, costs, delta):#, quant):
    """ Initializes the class attributes.

        Args:
              points: list, contains all individual demands represented by their quantile
              weights: dict, contains number of pax for each demand
              capacity: int, capacity of one helicopter (homogenous for now)
              costs: dict, cost of pooling any pair (i, j) of individual demand
              delta: int, maximum spread



        Returns:
                None

    """
    self.points = points
    self.weights = weights
    self.capacity = capacity
    self.groups = groups
    self.costs = costs
    self.delta = delta
    #self.quant = quant


  def build_model(self):
    """ Build the Integers Linear Program instance.
        Define the variables, objective function and the linear constraints.
      --------------
      Params :
              None
      --------------
      Returns :
              None

    """
    # First creates the master problem variables of whether
    self.x = plp.LpVariable.dicts("x", (self.points, self.groups), 0, 1, plp.LpInteger)

    #Create auxiliary variable to monitor demands pooled together
    self.tog = plp.LpVariable.dicts("together", (self.points, self.points, self.groups), 0, 1, plp.LpInteger)

    #Instantiate the model
    self.model = plp.LpProblem(name="optim", sense=plp.LpMinimize)

    #Objective function

    self.model += plp.lpSum([self.tog[i][j][k] * self.costs[(i, j)] for k in self.groups for i in self.points for j in self.points]), "Total Costs"



    #Adding constraints
    for i in self.points:
        self.model += plp.lpSum([self.x[i][k] for k in self.groups]) == 1
        #constraining search
        for j in self.points:
            for k in self.groups:
                if self.costs[(i, j)] > self.delta:
                   self.model += self.tog[i][j][k] == 0

    for k in self.groups:
        self.model += plp.lpSum([self.x[i][k] * self.weights[i] for i in self.points]) <= self.capacity
        for i in self.points:
            for j in self.points:
                if i != j:
                    self.model += self.tog[i][j][k] <= self.x[i][k]
                    self.model += self.tog[i][j][k] <= self.x[j][k]
                    self.model += self.tog[i][j][k] >= 1 - (1 - self.x[i][k] + 1 - self.x[j][k])
                    self.model += self.tog[i][j][k] * self.costs[(i, j)] <= self.delta



    print("Model Built")
    print("----------")

  def solve(self, max_time, opt_gap, verbose=1):
    """ Solve the Integers Linear Program instance built in self.build_model().

      Args :
              max_time : int, maximum running time required in seconds.
              opt_gap : float, in (0, 1), if max_time is None, then the objective value
              of the solution is guaranteed to be at most opt_gap % larger than the true
              optimum.
              verbose : 1 to print log of resolution. 0 for nothing.

      Returns :
              Status of the model : Infeasible or Optimal.
                Infeasible indicates that all constraints could not be met.
                Optimal indicates that the model has been solved optimally.

    """
    start = time.time()
    self.model.solve(plp.PULP_CBC_CMD(maxSeconds = max_time, fracGap = opt_gap, msg = verbose, mip_start=False, options=["randomCbcSeed 31"]))
    #Get Status
    print("Status:", plp.LpStatus[self.model.status])
    print("Total Costs = ", plp.value(self.model.objective))
    print("Solving time : ", round(time.time() - start, 3), " seconds.")
    return plp.LpStatus[self.model.status]


