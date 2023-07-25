Simulation work flow: 

First create the following file structure, so data will be saved correctly:
SomeMainFolder/ -- DataOutput
                -- FolderWhereSimulationFilesAre

multiSimHandler.py --> set all simulation parameters except number of users
  in function load_params. To run a simulation with these parameters, run
  "python multiSimHandler.py"

  Simulation data will be output in a timestamped folder in ../DataOutput
  To generate plots as in: 
    Figure 2: call plot_convergence_study, with the folder names of the three data sets you want to use
      as arguments.
    Figure 3: call plot_TR_from_text, with the folder names of the single data set you want to use as argument.
    Figure 4: call plot_max_min_diff, with the 6 data sets you want to use as arguments. Ideally, these shouold
      be structured as pairs, with 1 and 2, 3 and 4, and 5 and 6 being data sets from simulations that only differ in
      how the user max rates were set.

  Simulation parameters are:
    iters: number of time slots to run each simulation for
    runs: number of times to independently run the simulation. Final data output
      will average over the independent runs

    Nexcl: legacy parameter, not needed, was at some point for excluding the specified
      number of data points from plotting and some analysis
    H_num: Number of resource nodes to equipe the EGS with
    p_gen: probability of entanglement generation succeeding, when a communication session
      is allocated one resource for one time slot.
    
    global_scale: parameter used in specifying the minimum rates set by all communication sessions.
      In the example these parameters are set uniformly by all communication sessions as
      p_gen / global_scale. To set these parameters non-uniformly, a few short lines of code 
      could be written. See how the user max rates are being defined for an example.

    max_sched_per_q: fixes globally for all communication sessions the parameter x_s, the maximum number of
      resources that can be allocated to communication session s in a single time slot. To set this parameter non
      -uniformly, again a few short lines of code should be written

    sessionSamples: to take a random fraction of all the possible (N choose 2) communication sessions and run all 
      runs of the simulation with the same random sample of sessions, set this parameter.

    user_max_rates: set the parameters \lambda_u \forall u. set this by choosing one of 5 keywords. 
      "uniformVeryHigh": all u set [((NQs - 1) / 2) * p_gen] * NumUsers
      "halfUniformSessionMax": all u set [p_gen * max_sched_per_q / 2] * NumUsers
      "singleNonUniformSessionMax": randomly partition u in two groups. Group 1 is 1/4 of all 
        {u} and sets (p_gen * max_sched_per_q) / 2. Group 2 is the rest of {u} and sets p_gen * max_sched_per_q
      "doubleNonUniformSessionMax": randomly partition {u} in three groups. Group 1 is 1/4 of all {u} and 
        sets (p_gen * max_sched_per_q) / 2. Group 2 is another 1/4 of all {u} and sets (p_gen * max_sched_per_q) * 1.5.
        Remaining 1/2 of all {u} set p_gen * max_sched_per_q.

    step_size: changes the stepsize used in simulation. Should only be changed in accordance with the upper bound on step
      size in the RCP Convergence Theorem, Theorem 3.1

    param_change: boolean, fixes whether to change the simulation parameters at periodic intervals 
    change_key: if changing parameters, specifies how to change them. Currently supported is "ChangeH" or False.
      "ChangeH" will modify the numbr of resources available at the EGS periodically. 
    changes: sets the number of times to introduce changes. Number of iters will be divided by changes, creating 
      blocks of time steps. At the end of time step block a change will be made. 

  NumUsers: set this at the bottom of the file, in main. Sets N, the number of nodes/users connected to the EGS. 
