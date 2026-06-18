rWald <- function(n, B, v, A)
  # random deviate function for single acumulator
{
  rwaldt <- function(n, k, l, tiny = 1e-6) {
    # random sample of n from a Wald (or Inverse Gaussian)
    # k = criterion, l = rate, assumes sigma=1 Browninan motion
    # about same speed as statmod rinvgauss
    
    rlevy <- function(n = 1, m = 0, c = 1) {
      if (any(c < 0))
        stop("c must be positive")
      c / qnorm(1 - runif(n) / 2) ^ 2 + m
    }
    
    flag     <- l > tiny
    x        <- rep(NA, times = n)
    x[!flag] <- rlevy(sum(!flag), 0, k[!flag] ^ 2)
    mu       <- k / l
    lambda   <- k ^ 2
    y        <- rnorm(sum(flag)) ^ 2
    mu.0     <- mu[flag]
    lambda.0 <- lambda[flag]
    x.0      <- mu.0 + mu.0 ^ 2 * y / (2 * lambda.0) -
      sqrt(4 * mu.0 * lambda.0 * y + mu.0 ^ 2 * y ^ 2) * mu.0 /
      (2 * lambda.0)
    
    z             <- runif(length(x.0))
    test          <- mu.0 / (mu.0 + x.0)
    if(any(is.na(test))){
      print(paste0('l = ',l))
      print(paste0('k = ',k))
    }
    x.0[z > test] <- mu.0[z > test] ^ 2 / x.0[z > test]
    x[flag]       <- x.0
    x[x < 0]      <- max(x)
    x
  }
  
  # Act as if negative v never terminates, cluge to do single accumulator
  # case by passing negative v
  if (length(v) != n)
    v <- rep(v, length.out = n)
  if (length(B) != n)
    B <- rep(B, length.out = n)
  if (length(A) != n)
    A <- rep(A, length.out = n)
  
  # Kluge to return -Inf for negative rates, so can implment one accumulator case
  out      <- numeric(n)
  ok       <- v > 0
  nok      <- sum(ok)
  bs       <- B[ok] + runif(nok, 0, A[ok])
  out[ok]  <- rwaldt(nok, k = bs, l = v[ok])
  out[!ok] <- Inf
  out
}