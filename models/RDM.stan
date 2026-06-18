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
  real B_raw;  // Boundary height
  real b_raw;  // Response bias
  real V_raw;  // Response speed
  real v_raw;  // Response quality
  real s_raw;  // Variance difference
  real t0_raw; // non-decision time
}
transformed parameters{
  real b_min1      = exp(B_raw - 0.5 * b_raw);
  real b_pos1      = exp(B_raw + 0.5 * b_raw);
  real v_correct   = exp(V_raw + 0.5 * v_raw);
  real v_incorrect = exp(V_raw - 0.5 * v_raw);
  real s_correct   = exp(1 + s_raw);
  real s_incorrect = exp(1 - s_raw);
  real t0          = Phi(t0_raw) * min_rt;
  
  // Precompute trial-invariant parameters
  vector[2] v_pos1_s;
  vector[2] v_min1_s;
  vector[2] b_pos1_s;
  vector[2] b_min1_s;

  // For stim == 1
  v_pos1_s[1] = v_correct / s_correct;
  v_min1_s[1] = v_incorrect / s_incorrect;
  b_pos1_s[1] = b_pos1 / s_correct;
  b_min1_s[1] = b_min1 / s_incorrect;

  // For stim == 2
  v_pos1_s[2] = v_incorrect / s_incorrect;
  v_min1_s[2] = v_correct / s_correct;
  b_pos1_s[2] = b_pos1 / s_incorrect;
  b_min1_s[2] = b_min1 / s_correct;
}
model{
  B_raw  ~ normal(0,2);
  b_raw  ~ std_normal();
  V_raw  ~ normal(0,2);
  v_raw  ~ std_normal();
  s_raw  ~ std_normal();
  t0_raw ~ std_normal();
  
  for (n in 1:N) {
    if (yi[n] == 1) {
      real t = y[n] - t0;
      int s_idx = stim[n]; // 1 or 2
      
      // Fast lookup via precomputed arrays and direct selection based on choice
      if (choice[n] == 1) {
        target += wald_lpdf(t  | v_pos1_s[s_idx], b_pos1_s[s_idx]);
        target += wald_lccdf(t | v_min1_s[s_idx], b_min1_s[s_idx]);
      } else {
        target += wald_lpdf(t  | v_min1_s[s_idx], b_min1_s[s_idx]);
        target += wald_lccdf(t | v_pos1_s[s_idx], b_pos1_s[s_idx]);
      }
    }
  }
}
