## How to test the MC code for the DNN? 

#### Proj: Yoshino_4.0



### The Problem: 

Since the version `Yoshino_3.0`, we simplified the calculation of the MC move. In other words, we replace calculating all total energy for all gaps by calculating the energy change change $\Delta E$ only induced by the activated gaps. But when we compare the results between version `Yoshino_2.0` and the version after 3.0, we find the calculated Overlaps for $J$ and $S$ are very different. Therefore, to make sure our algorithm is correct, we have to test the code. How to test the two codes?  We can use pre-generated random series for tuple $(\mu,l,n)$ and for $(l,n_2,n_1)$.  Is it enough? In other words,  is it enough to make sure two MC simulations from the same initial configurations will produce the same result? No, because when we update $J$, we still introduce randomness!

Therefore, to make sure two MC simulations from the same initial configurations will produce the same result, we need a four-tuple $(l,n_2,n_1,x)$, where $x$ is a random number following the Gaussian distribution with zero mean and variance 1. Ref. to Yoshino2020. Is that fine? Looks good, but, notice that in the tuple $(l,n_2,n_1,x)$, all the elements are integers except the last one. Therefore, it is not convenient to append as a list! Therefore, instead use a four-tuple random series for updating $J$, we use a  pre-generated random series of tuple $(l,n_2,n_1)$ and a list of $x$.

Is it enough to produce exactly same results? NO! Because in MC algorithm, we also introduced randomness! Therefore, we need an extra series for store these random numbers.  Actually, to make it easier to implement, we also create two series of random numbers, corresponding to updating $S$ and updating $J$.

### Solution:

The Steps:

Step 1: First, we generate $P+1$ series $Series_S0$,$Series_S1$, $Series_S2$, $Series_S3$,...,$Series_SP$, for random tuple $(\mu,l,n)$ 
and P+1 series $Series_J0$,...,$Series_JP$, for random tuple $(l,n_2,n_1)$ in a function `host_as_generator.py`. 



Step 2: At the same time, we run MC simulation with the series pair $Series_S0$ and $Series_J0$, and the output of the MC simulation is $S_{traj}$  and $J_{traj}$ for the host machine. In the simulation, we have to save the initial states of $S$ and $J$  for the DNN model. And we 

assume that all the simulations in guest machine start from the same initial states of $S$ and $J$. 



Step 3: 3a-3d can be run parallelly.

Step 3a:  Then we will do not use the results in step 2 at the moment. In this step we run MC simulation with Yoshino_2.0 code, from $Series_S0$ and $Series_J0$ and the results $S_{traj}$ and $J_{traj}$ play the role of variable trajectory for psedo-host machine.  See `guest_old_test_as_host.py`.

Step 3b:  Then we will do not use the results in step 2 at the moment. In this step we run MC simulation with Yoshino_2.0 code, from $Series_S1$ and $Series_J1$ and the results $S_{traj,guest}$ and $J_{traj,guest}$ play the role of variable trajectory for psedo-guest machine. See `guest_old_test_as_guest.py`.

Step 3c:  Then we will do not use the results in step 2 at the moment. In this step we run MC simulation with Yoshino_4.0 code, from $Series_S0$ and $Series_J0$ and the results $S_{new,traj}$ and $J_{new,traj}$ play the role of variable trajectory for psedo-host machine.  See `guest_new_test_as_host.py`.

Step 3d:  Then we will do not use the results in step 2 at the moment. In this step we run MC simulation with Yoshino_4.0 code, from $Series_S1$ and $Series_J1$ and the results $S_{new,traj,guest}$ and $J_{new,traj,guest}$ play the role of variable trajectory for psedo-guest machine.   See `guest_new_test_as_guest.py`.



Step 4: 4a-4d can be run parallelly.

Step 4a: Calculate Overlap $q(t,l)$ for $S_{traj}$  and $S_{traj,guest}$.

Step 4b: Calculate Overlap $Q(t,l)$ for $ J_{traj}$  and $ J_{traj,guest}$.

Step 4c: Calculate Overlap $q_{new}(t,l)$ for $S_{new,traj}$  and $S_{new,traj,guest}$.

Step 4d: Calculate Overlap $Q_{new}(t,l)$ for $ J_{new,traj}$  and $ J_{new,traj,guest}$.



Step 5: 5a-5b can be run parallelly.

Step 5a: Compare $q(t,l)$ and $q_{new}(t,l)$, if they are not exactly the same, then one of our code may be wrong. If they are exactly the same, we set $correct1 = 1$.

Step 5b: Compare $ Q(t,l)$ and $ Q_{new}(t,l)$, if they are not exactly the same, then one of our code may be wrong, too.  If they are exactly the same, we set $correct2 = 1$.



Step 6: If both $correct1$ and $correct2$ are equal to 1, our MC algorithms are correct.

