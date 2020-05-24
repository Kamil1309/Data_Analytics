data {
  real alpha;
  real mean_1; // Cant use key word 'mean'
  real phi;
}


generated quantities {
    
    real neg_bin_con = neg_binomial_rng(alpha, alpha/mean_1);
    real neg_bin_dis = neg_binomial_2_rng(mean_1, phi);
    
    real poisson_con = poisson_rng(neg_bin_con);
    real poisson_dis = poisson_rng(neg_bin_dis);
    
}