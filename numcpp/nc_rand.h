#pragma once

namespace nc{
//* ----nc_rand----
double generate_gaussian_noise(
  std::uniform_int_distribution<uint32_t> & dist,
  std::mt19937& rng, double mean, double stddev);

vector<double> randn(  std::uniform_int_distribution<uint32_t> & dist,
  std::mt19937& rng, int len_noise, double mean, double stddev);
}