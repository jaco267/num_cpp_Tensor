#include "numcpp.h"
#include <random>
namespace nc{

double generate_gaussian_noise(
  std::uniform_int_distribution<uint32_t> & dist,
  std::mt19937& rng, double mean, double stddev) {
  // Generate two uniformly distributed random numbers between 0 and 1
  double i0 = static_cast<double>(dist(rng));
  double i1 =  static_cast<double>(dist(rng));
  double u1 = i0 / E2MAX;
  double u2 = i1 / E2MAX;

  // Apply Box-Muller transform
  double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);

  return z0 * stddev + mean;
}
vector<double> randn(  std::uniform_int_distribution<uint32_t> & dist,
  std::mt19937& rng, int len_noise, double mean, double stddev){
  vector<double> noise; 
  noise.resize(len_noise,0); 
  for (int i=0; i< len_noise; i++){
    noise[i] = generate_gaussian_noise(dist,rng,mean, stddev);
  }
  return noise; 
}
}