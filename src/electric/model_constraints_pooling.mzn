
%-- reshaping data demands
array[1..n_demands*6] of float: demands_info_1d;
array[0..n_demands-1, 0..6-1] of float: demands_info = array2d(0..n_demands-1, 0..6-1, demands_info_1d);


set of int: demands_ids = 0..n_demands-1;

% -- Declaring variables
array[demands_ids, demands_ids] of var bool: x; % assignement
array[demands_ids, demands_ids] of var 0.0..720.0: a; % expected waiting times
array[demands_ids] of var bool: y; % bin usage
array[demands_ids] of var 0.0..720.0: f; % flight departure time




% -- Constraints

% Symetry breaking
constraint forall (k in 0..n_demands-2) (
                                         bool2int(y[k+1]) <= bool2int(y[k]) /\ f[k+1] <= f[k]

);

% Aircraft capacity constraints and activation
constraint forall (k in demands_ids) (
                                        sum (i in demands_ids) (x[i, k] * demands_info[i, NPAX]) <= capacity * y[k]

);


% Each demand must be assigned to a group
constraint forall (i in demands_ids) (
                                        sum (k in demands_ids) (x[i, k]) = 1

);


% Maximum departure time must be respected
constraint forall (k in demands_ids, i in demands_ids) (
                                                          x[i, k] -> f[k] <= demands_info[i, MAX_DEP]

);

% Maximum expected waiting time - dependent on passenger class
constraint forall (k in demands_ids, i in demands_ids) (
                                                          x[i, k] -> f[k] - demands_info[i, MEAN_ARRIVAL] <= demands_info[i, CLASS] * max_wait_premium + (1 - demands_info[i, CLASS]) * max_wait_regular

);


% Robust flight departure time

constraint forall (k in demands_ids, i in demands_ids) (
                                                        f[k] >= x[i, k] * demands_info[i, QUANT]

);


constraint forall (k in demands_ids, i in demands_ids) (
                                                        a[i, k] = if x[i, k]
                                                                  then
                                                                  (f[k] - demands_info[i, MEAN_ARRIVAL]) * (demands_info[i, CLASS] * alpha_premium + (1 - demands_info[i, CLASS]) * alpha_regular)
                                                                  else
                                                                  0
                                                                  endif

);

%-- Define objective

% -- Calling minimize on objective function

var float: usage = sum (k in demands_ids) (
                                            bool2int(y[k]) * lbda_p

                                           );


var float: overall_waiting =   sum (k in demands_ids, i in demands_ids) (
                                                                        a[i, k]
                                                                         );

var float: obj = usage + overall_waiting;


solve minimize obj;