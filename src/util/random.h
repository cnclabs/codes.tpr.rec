#ifndef RANDOM_H
#define RANDOM_H

void shuffle_the_vector(std::vector<long>& input_vector);
double random_range(const long& min, const long& max);
double random_prob();
double ran_uniform();
double ran_gaussian();
double ran_gaussian(double mean, double stdev);

#endif
