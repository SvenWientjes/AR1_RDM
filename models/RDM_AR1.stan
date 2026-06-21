functions{
  // Compute wald lpdf
  real wald_lpdf(real t, real v, real b) {
    if(t<=1e-8) return negative_infinity();
    real log2pi = 0.9189385;
    real log_t_term = -1.5 * log(t);
    real common_term = log(b) + log_t_term - log2pi;

    if (v < 1e-8) {
        // Zero drift: Lévy distribution
        return common_term - (b * b) / (2 * t);
    } else {
        real exponent = - square(b - v * t) / (2 * t);
        return common_term + exponent;
    }
  }

  // Complementary cumulative wald lpdf
  real wald_lccdf(real t, real v, real b) {
    if(t<=1e-8) return 0;
    if (v < 1e-8) return chi_square_lcdf((pow(b,2) / t) | 1);
  
    real sqrt_t = sqrt(t);
    //real z1 = (v*t-b);
    //real z2 = (-(v*t+b));
    real log_phi1 = normal_lcdf(v*t-b|0,sqrt_t);
    real log_phi2 = normal_lcdf(-(v*t+b)|0,sqrt_t);

    // Standard Wald CDF, in log-space:
    real logF = log_sum_exp(log_phi1, 2 * b * v + log_phi2);
    return log1m_exp(logF);
  }
}
data {
  int<lower=1> N;                       // Number of trials
  array[N] int<lower=1,upper=2> stim;   // stimulus (1 or 2)
  array[N] int<lower=1,upper=2> choice; // choice (1 or 2)
  array[N] int<lower=0,upper=1> yi;     // Trial inclusion (0:no, 1:yes)
  vector[N] y;                          // RTs (s)
  real<lower=0> min_rt;                 // Minimum observed RT (to constrain t0)
}
parameters{
  // Mean-level parameters
  real B_raw;  // Boundary height
  real b_raw;  // Response bias
  real V_raw;  // Response speed
  real v_raw;  // Response quality
  real s_raw;  // Drift rate variability
  real t0_raw; // non-decision time
  
  // AR(1) persistence
  real rho_B_raw;
  real rho_b_raw;
  real rho_V_raw;
  real rho_v_raw;

  // Innovation scales
  real log_sigma_B;
  real log_sigma_b;
  real log_sigma_V;
  real log_sigma_v;

  // Non-centered standard normal innovations
  vector[N] B_z;
  vector[N] b_z;
  vector[N] V_z;
  vector[N] v_z;
}
transformed parameters{
  
  // Correlations of AR(1) process
  real rho_B = inv_logit(rho_B_raw);
  real rho_b = inv_logit(rho_b_raw);
  real rho_V = inv_logit(rho_V_raw);
  real rho_v = inv_logit(rho_v_raw);

  // Variance of AR(1) process
  real<lower=0> sigma_B = exp(log_sigma_B);
  real<lower=0> sigma_b = exp(log_sigma_b);
  real<lower=0> sigma_V = exp(log_sigma_V);
  real<lower=0> sigma_v = exp(log_sigma_v);

  vector[N] B_t;
  vector[N] b_t;
  vector[N] V_t;
  vector[N] v_t;

  matrix[N, 2] v_acc;
  matrix[N, 2] b_acc;
  
  // Non-decision time
  real t0 = Phi(t0_raw) * min_rt;

  {
    // Initialize deviations
    vector[N] B_dev;
    vector[N] b_dev;
    vector[N] V_dev;
    vector[N] v_dev;

    // Precompute constants for the AR(1) processes
    real scale_B = sqrt(1 - square(rho_B));
    real scale_b = sqrt(1 - square(rho_b));
    real scale_V = sqrt(1 - square(rho_V));
    real scale_v = sqrt(1 - square(rho_v));

    // Seed starting deviation
    B_dev[1] = B_z[1];
    b_dev[1] = b_z[1];
    V_dev[1] = V_z[1];
    v_dev[1] = v_z[1];

    // Loop over vector of deviations
    for (n in 2:N) {
      B_dev[n] = rho_B * B_dev[n - 1] + scale_B * B_z[n];
      b_dev[n] = rho_b * b_dev[n - 1] + scale_b * b_z[n];
      V_dev[n] = rho_V * V_dev[n - 1] + scale_V * V_z[n];
      v_dev[n] = rho_v * v_dev[n - 1] + scale_v * v_z[n];
    }
  
    // Non-centered mapping to true parameter values over time
    B_t = B_raw + sigma_B * B_dev;
    b_t = b_raw + sigma_b * b_dev;
    V_t = V_raw + sigma_V * V_dev;
    v_t = v_raw + sigma_v * v_dev;

    // --- ARRAYS FOR QUICK LOOKUP BY ACCUMULATOR ---
    // Column 1 = Accumulator 1 (pos1), Column 2 = Accumulator 2 (min1)
    {
      vector[N] v_correct   = exp(V_t + 0.5 * v_t);
      vector[N] v_incorrect = exp(V_t - 0.5 * v_t);
      
      vector[N] b_pos1 = exp(B_t + 0.5 * b_t);
      vector[N] b_min1 = exp(B_t - 0.5 * b_t);
      
      real s_correct   = exp(1 + s_raw);
      real s_incorrect = exp(1 - s_raw);
  
      for(n in 1:N) {
        if(stim[n] == 1) {
          v_acc[n, 1] = v_correct[n] / s_correct;
          v_acc[n, 2] = v_incorrect[n] / s_incorrect;
          b_acc[n, 1] = b_pos1[n] / s_correct;
          b_acc[n, 2] = b_min1[n] / s_incorrect;
        } else { // stim == 2
          v_acc[n, 1] = v_incorrect[n] / s_incorrect;
          v_acc[n, 2] = v_correct[n] / s_correct;
          b_acc[n, 1] = b_pos1[n] / s_incorrect;
          b_acc[n, 2] = b_min1[n] / s_correct;
        }
      }
    }
  }
}
model {
  // Priors
  B_raw  ~ normal(0, 2);
  b_raw  ~ std_normal();
  V_raw  ~ normal(0, 2);
  v_raw  ~ std_normal();
  s_raw  ~ std_normal();
  t0_raw ~ std_normal();
  
  rho_B_raw ~ normal(0, 0.5);
  rho_b_raw ~ normal(0, 0.5);
  rho_V_raw ~ normal(0, 0.5);
  rho_v_raw ~ normal(0, 0.5);

  log_sigma_B ~ normal(-1.203973,0.5);
  log_sigma_b ~ normal(-1.203973,0.5);
  log_sigma_V ~ normal(-1.203973,0.5);
  log_sigma_v ~ normal(-1.203973,0.5);

  B_z ~ std_normal();
  b_z ~ std_normal();
  V_z ~ std_normal();
  v_z ~ std_normal();
  
  // Likelihood loop
  for (n in 1:N) {
    if (yi[n] == 1) {
      real t = y[n] - t0;
      int c = choice[n];       // 1 or 2
      int alt_c = 3 - choice[n]; // Alternate accumulator (if c=1 then 2, if c=2 then 1)
      
      target += wald_lpdf(t | v_acc[n, c], b_acc[n, c]);
      target += wald_lccdf(t | v_acc[n, alt_c], b_acc[n, alt_c]);
    }
  }
}
