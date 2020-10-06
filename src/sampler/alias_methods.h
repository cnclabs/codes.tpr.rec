#ifndef ALIAS_METHOD_H
#define ALIAS_METHOD_H
#include <vector>
#include <cmath>
#include "../util/random.h"

class AliasMethods {
    /* AliasMethod is an efficient implementation of weighted sampling
     * Reference: https://en.wikipedia.org/wiki/Alias_method
     * In this code, we modify it by concating multiple alias blocks into one.
     * Ask CM (changecandy@gmail.com) for more details.
     */
    public:
        // constuctor
        AliasMethods();

        // variables
        std::vector<long> offset, branch;
        std::vector<long> alias_position;
        std::vector<double> alias_probability;

        // append a new distribution with alias method
        void append(std::vector<double>& distribution, const double power);

        // functions
        long draw();
        long draw_uniformly();
        long draw(long index);
        long draw_safely(long index);
        long get_offset(long node);
        long get_branch(long node);
};
#endif
