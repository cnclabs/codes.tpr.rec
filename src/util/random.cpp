using namespace std;
#include <algorithm>
#include <random>
#include <limits.h>
#include "random.h"

void shuffle_the_vector(vector<long>& input_vector) {
    static thread_local random_device rd;
    static thread_local mt19937 generator(rd());
    shuffle(input_vector.begin(), input_vector.end(), generator);
}

double random_range(const long & min, const long & max) {
    static thread_local random_device rd;
    static thread_local mt19937* generator = new mt19937( rd() );
    uniform_real_distribution<double> distribution(min, max);
    return distribution(*generator);
}

double random_prob() {
    static thread_local random_device rd;
    static thread_local mt19937* generator = new mt19937( rd() );
    static thread_local uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(*generator);
}

double ran_uniform() {
    return rand()/((double)RAND_MAX + 1);
}

double ran_gaussian() {
    // Joseph L. Leva: A fast normal Random number generator
    double u,v, x, y, Q;
    do {
        do {
            u = ran_uniform();
        } while (u == 0.0);
        v = 1.7156 * (ran_uniform() - 0.5);
        x = u - 0.449871;
        y = std::abs(v) + 0.386595;
        Q = x*x + y*(0.19600*y-0.25472*x);
        if (Q < 0.27597) { break; }
    } while ((Q > 0.27846) || ((v*v) > (-4.0*u*u*std::log(u))));
    return v / u;
}

double ran_gaussian(double mean, double stdev) {
    if ((stdev == 0.0) || (std::isnan(stdev))) {
        return mean;
    } else {
        return mean + stdev*ran_gaussian();
    }
}

