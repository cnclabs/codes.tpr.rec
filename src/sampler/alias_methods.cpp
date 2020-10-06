#include "alias_methods.h"
#include <iostream>

AliasMethods::AliasMethods() {
}

void AliasMethods::append(std::vector<double>& distribution, const double power) {

    long offset = this->alias_position.size();
    long branch = distribution.size();
    this->offset.push_back(offset);
    this->branch.push_back(branch);

    // normalization of vertices weights
    double sum, norm;
    std::vector<double> norm_prob;

    sum = 0;
    for (auto weight: distribution)
    {
        sum += pow(weight, power);

        // get space at the same time
        this->alias_position.push_back(-1);
        this->alias_probability.push_back(1.1); // any value > 1.0
    }
    norm = distribution.size()/sum;

    for (auto weight: distribution)
    {
        norm_prob.push_back( pow(weight, power)*norm );
    }

    // block divison
    std::vector<long> small_block, large_block;

    for (long pos=0; pos!=norm_prob.size(); ++pos)
    {
        if ( norm_prob[pos]<1 )
        {
            small_block.push_back( pos );
        }
        else
        {
            large_block.push_back( pos );
        }
    }

    // assign alias table
    long small_pos, large_pos;

    while (small_block.size() && large_block.size())
    {
        small_pos = small_block.back();
        small_block.pop_back();
        large_pos = large_block.back();
        large_block.pop_back();

        this->alias_position[offset+small_pos] = offset+large_pos;
        this->alias_probability[offset+small_pos] = norm_prob[small_pos];
        norm_prob[large_pos] = norm_prob[large_pos] + norm_prob[small_pos] - 1;
        if (norm_prob[large_pos] < 1)
        {
            small_block.push_back( large_pos );
        }
        else
        {
            large_block.push_back( large_pos );
        }
    }

    while (large_block.size())
    {
        large_pos = large_block.back();
        large_block.pop_back();
    }

    while (small_block.size())
    {
        small_pos = small_block.back();
        small_block.pop_back();
    }
}

long AliasMethods::draw_uniformly() {
    return random_range(0, this->alias_position.size());
}

long AliasMethods::draw() {

    long sample_position = random_range(0, this->alias_position.size());
    double sample_probability = random_prob();

    if (sample_probability < this->alias_probability[sample_position])
        return sample_position;
    else
        return this->alias_position[sample_position];
}

long AliasMethods::get_offset(long index) {
    return this->offset[index];
}

long AliasMethods::get_branch(long index) {
    return this->branch[index];
}

long AliasMethods::draw(long index) {

    long sample_position = this->offset[index] + random_range(0, this->branch[index]);
    double sample_probability = random_prob();

    if (sample_probability < this->alias_probability[sample_position])
        return sample_position;
    else
        return this->alias_position[sample_position];
}

long AliasMethods::draw_safely(long index) {

    if (this->branch[index]==0)
        return -1;

    long sample_position = this->offset[index] + random_range(0, this->branch[index]);
    double sample_probability = random_prob();

    if (sample_probability < this->alias_probability[sample_position])
        return sample_position;
    else
        return this->alias_position[sample_position];
}
