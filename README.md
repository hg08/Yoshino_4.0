
Tutorial for the project ```hierarchical_free_energy_landscape_in_DNN```

1. One must first run  

```host.py```

before run  

```guest.py```

since the guest machine in the code `guest.py` have to use the same initial configuration as the host machine in `host.py`. The initial configurations will be produced by 
the code `host.py` automatically, and they will be used in `guest.py` when one run `guest.py`.

2. Yoshino_3.0 is based on the following basic ideas: 

1) When calculate the $\Delta H_{eff}$, we just need to consider the sites directly related to the active sites (bonds). 

2) Depend on the types of active object ( $J$ or $S$ ), we need different update scheme for $\Delta H_{eff}$.  If $S$ is activated, then we only calculate the $\Delta H_{eff}$ caused by this single machine; but is J is activated, $J$ in all $M$ machines are changed.

3. Yoshino_4.0 is based on the following basic ideas:
1) S (J) at Every other layer can be updated at the same time.
2) The order of updating $J$ or $S$ is not important, therefore, in Yoshino_4.0, we remove the random choice of $J$ or $S$ before updating.
