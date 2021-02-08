%--- Adding time compatible to reduce number of constraints instanciated.
array[1..n_requests*n_requests] of float: time_compatible_info_1d;
array[0..n_requests-1, 0..n_requests-1] of float: time_compatible_info = array2d(0..n_requests-1, 0..n_requests-1, time_compatible_info_1d);



% Calling table from data files
array[1..n_requests*n_requests] of float: costs_info_1d;
array[0..n_requests-1, 0..n_requests-1] of float: costs = array2d(0..n_requests-1, 0..n_requests-1, costs_info_1d);


array[1..n_requests*6] of int: requests_info_1d;
array[0..n_requests-1, 0..5] of int: requests_info = array2d(0..n_requests-1, 0..5, requests_info_1d);

array[1..n_skyports*n_skyports] of int: time_info_1d;
array[0..n_skyports-1, 0..n_skyports-1] of int: time_info = array2d(0..n_skyports-1, 0..n_skyports-1, time_info_1d);


array[1..n_skyports*n_skyports] of float: energy_info_1d;
array[0..n_skyports-1, 0..n_skyports-1] of float: energy_info = array2d(0..n_skyports-1, 0..n_skyports-1, energy_info_1d);


array[1..n_aircraft*3] of int: aircraft_info_1d;
array[0..n_aircraft-1, 0..2] of int: aircraft_info = array2d(0..n_aircraft-1, 0..2, aircraft_info_1d);





set of int: req_idx = 0..n_requests-1;
set of int: aircraft_idx = 0..n_aircraft-1;
% set of int: time_steps = 0..T-1;
set of int: locations = 0..n_skyports-1;
set of int: real_req = 0..n_requests-n_aircraft-1;
set of int: fake_req = n_requests-n_aircraft..n_requests-1;

% -- Defining variables
array[req_idx, req_idx, aircraft_idx] of var bool: y;
array[req_idx, req_idx, aircraft_idx] of var 0.0..domain_charge_float: ta;
array[req_idx, req_idx, aircraft_idx] of var 0.0..domain_charge_float: tb;
array[req_idx, req_idx, aircraft_idx] of var bool: sa;
array[req_idx, req_idx, aircraft_idx] of var bool: sb;
array[req_idx, req_idx, aircraft_idx] of var 0.0..100.0: ba;
array[req_idx, req_idx, aircraft_idx] of var 0.0..100.0: bb;
array[req_idx, req_idx, aircraft_idx] of var 0.0..200.0: b;
array[req_idx, aircraft_idx] of var 0.0..100.0: e_after;
array[req_idx, aircraft_idx] of var 0.0..100.0: e_before;
var int: unserved;
var int: fast_charges;

% -- Constraints

% Defining domain of definition for all variables and restrict b


constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
      b[r1, r2, v] = bb[r1, r2, v] + ba[r1, r2, v] %b is determined by ba and bb
);


% Computing number of unserved demands and fast charges
constraint unserved = sum (r2 in real_req) (
                                            1 -  (sum(r1 in req_idx where r1 != r2, v in aircraft_idx)  (y[r1, r2, v]))
                                            );

constraint fast_charges = sum (r1 in req_idx, r2 in req_idx where r1 != r2 /\ time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                              bool2int(sa[r1, r2, v]) + bool2int(sb[r1, r2, v])
                                              );
% Do not use diagonal
constraint forall(r1 in req_idx, v in aircraft_idx) (not(y[r1,r1,v]));

% Logical constraint on service assigment : unicity & continuity

% At most one aircraft serves r1 and r2 sequentially, in that order
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0) (
                                                  sum (v in aircraft_idx) (y[r1, r2, v]) <= 1
                                                  );

% At most one arcs comes into r2, forall r2
constraint forall (r2 in req_idx) (
                                  sum (r1 in req_idx, v in aircraft_idx) (y[r1, r2, v]) <= 1
                                  );

% At most one arcs leaves r1
constraint forall (r1 in req_idx) (
                                     sum (r2 in req_idx, v in aircraft_idx) (y[r1, r2, v]) <= 1
                                   );

% If r1, r2 is served by v, then there exist p such that v serves p, r1.
constraint forall (r1 in real_req, r2 in real_req where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                       y[r1, r2, v] <= sum (p in req_idx where p != r1) (y[p, r1, v])

                                                      );

% -- Preventing to serve fake request not attributed to v
constraint forall (v in aircraft_idx, r2 in req_idx, r1 in req_idx where r1 >= n_requests - n_aircraft /\ r1 != n_requests - n_aircraft + v) (

                                                            not(y[r1, r2, v])

);



% Prevents from going in the past- Enforce valid connections

% constraint forall (r1 in req_idx, r2 in req_idx, v in aircraft_idx) (
%                                                                       y[r1, r2, v] ->

%                                                                       if requests_info[r1, dest_location] = requests_info[r2, source_location]

%                                                                       then

%                                                                       requests_info[r1, dest_time] + delta < requests_info[r2, source_time]

%                                                                       else

%                                                                       requests_info[r1, dest_time] + time_info[requests_info[r1, dest_location], requests_info[r2, source_location]] + 2 * delta < requests_info[r2, source_time]

%                                                                       endif
%                                                                      );

constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 0.0, v in aircraft_idx) (
                                                                      not(y[r1, r2, v])
);

% % Forces the charging mode to be the same if there is no deadhead : you may choose charging mode once per landing.
% % --

constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                     requests_info[r2, dest_location] = requests_info[r2, source_location] -> sa[r1, r2, v] = sb[r1, r2, v]
                                                                     );

% % Charging times constraints

% Maximum charging time after serving r1, corresponding to latest possible departure from r1
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                      ta[r1, r2, v] <= max(requests_info[r2, source_time] - requests_info[r1, dest_time] -
                                                                      time_info[requests_info[r1, dest_location], requests_info[r2, source_location]] -
                                                                      delta * (requests_info[r1, dest_location] != requests_info[r2, source_location]), 0)
                                                                     );

% Maximum charging time before r2, corresponding to earliest possible arrival at r2
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                      tb[r1, r2, v] <= max(requests_info[r2, source_time] - requests_info[r1, dest_time] -
                                                                      time_info[requests_info[r1, dest_location], requests_info[r2, source_location]] -
                                                                      delta * (requests_info[r1, dest_location] != requests_info[r2, source_location]), 0)
                                                                     );

% Maximum combined charging time does not exceed idle time, i.e. free time minus deadhead time.
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                      tb[r1, r2, v] + ta[r1, r2, v] <= max(requests_info[r2, source_time] - requests_info[r1, dest_time] -
                                                                      time_info[requests_info[r1, dest_location], requests_info[r2, source_location]], 0)
                                                                     );


% % Computing electricity bought given charging times

% % First electricity bought after service of r1
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                      ba[r1, r2, v] = if sa[r1, r2, v] then
                                                                                        min(soc_high - e_after[r1, v],
                                                                                        gamma_f * ta[r1, r2, v])
                                                                                        else
                                                                                        min(soc_high - e_after[r1, v],
                                                                                        gamma_s * ta[r1, r2, v])
                                                                                        endif


                                                                     );

constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                      bb[r1, r2, v] = if sb[r1, r2, v] then
                                                                                        min(soc_high - (e_after[r1, v] + ba[r1,r2,v]
                                                                                                           - energy_info[requests_info[r1, dest_location], requests_info[r2, source_location]] ),
                                                                                                          gamma_f * tb[r1, r2, v])
                                                                                        else
                                                                                        min(soc_high - (e_after[r1, v] + ba[r1,r2,v]
                                                                                                           - energy_info[requests_info[r1, dest_location], requests_info[r2, source_location]] ),
                                                                                                          gamma_s * tb[r1, r2, v])
                                                                                        endif


                                                                     );

% % -- Monitoring evtols energy



% % Initial energy
constraint forall (v in aircraft_idx) (
                                       e_after[n_requests - n_aircraft + v, v] = soc_high
                                      );


% Energy after service
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                                    y[r1, r2, v] -> e_after[r2, v] =
                                                                                                    e_before[r2, v] -
                                                                                                    energy_info[requests_info[r2, source_location], requests_info[r2, dest_location]]
                                                                                   );

% % Energy after deadhead right before second service
constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                                    y[r1, r2, v] -> e_before[r2, v] =
                                                                                                    e_after[r1, v] -
                                                                                                    energy_info[requests_info[r1, dest_location], requests_info[r2, source_location]] + b[r1,r2,v]
                                                                                   );

% % Implement actual takeoff constraint : charge level must be at least soc_min to takeoff

constraint forall (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                                    y[r1, r2, v] -> e_before[r2, v] >= min_soc /\
                                                                                                    e_after[r1, v] + ba[r1,r2,v] >= min_soc
                                                                                   );





% -- Calling minimize on objective function

var float: op_cost = sum (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                                  y[r1, r2, v] * costs[r1, r2]);


var float: electricity =   sum (r1 in req_idx, r2 in req_idx where time_compatible_info[r1, r2] = 1.0, v in aircraft_idx) (
                                                                                  b[r1, r2, v] * pe);




var float: penalty_charges = pen_fast * fast_charges;


var float: penalty_unserved = pen_unserved * unserved;

var float: obj_float = op_cost + electricity + penalty_charges + penalty_unserved;



%ann: assignment_search = bool_search(array1d(y), dom_w_deg, indomain_min);
%ann: time_search = float_search(array1d(ta) ++ array1d(tb), 0.9, smallest, indomain_split);
%ann: charge_search = float_search(array1d(ba) ++ array1d(bb) ++ array1d(b), 0.9, input_order, indomain_min);
%ann: fast_charge_search = bool_search(array1d(sa) ++ array1d(sb), dom_w_deg, indomain_min);
%ann: e_search = float_search(array1d(e_before) ++ array1d(e_after), 0.9, smallest, indomain_min);
%ann: restart = restart_luby(100);
%ann: heuri_time = seq_search ([time_search, charge_search, e_search])::restart;
%ann: heuri_bool = seq_search ([assignment_search, fast_charge_search])::restart;
%ann: heuri_full = seq_search ([assignment_search, time_search, fast_charge_search, charge_search, e_search])::restart;


solve minimize obj_float;