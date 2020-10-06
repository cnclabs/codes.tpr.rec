#ifndef QUARDRUPLE_OPTIMIZER_H
#define QUARDRUPLE_OPTIMIZER_H
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

#define SIGMOID_TABLE_SIZE 1000
#define MAX_SIGMOID 8.0

class QuadrupleOptimizer {
    private:
        void init_sigmoid();

    public:
        // constructor
        QuadrupleOptimizer();

        // variables
        std::vector<double> cached_sigmoid;

        // functions
        double fast_sigmoid(double value);

        // loss
        void feed_trans_bpr_loss(std::vector<double>& from_embedding,
                                 std::vector<double>& relation_embedding,
                                 std::vector<double>& to_pos_embedding,
                                 std::vector<double>& to_neg_embedding,
                                 int dimension,
                                 std::vector<double>& from_loss,
                                 std::vector<double>& relation_loss,
                                 std::vector<double>& to_pos_loss,
                                 std::vector<double>& to_neg_loss);

        int feed_trans_margin_bpr_loss(std::vector<double>& from_embedding,
                                       std::vector<double>& relation_embedding,
                                       std::vector<double>& to_pos_embedding,
                                       std::vector<double>& to_neg_embedding,
                                       double margin,
                                       int dimension,
                                       std::vector<double>& from_loss,
                                       std::vector<double>& relation_loss,
                                       std::vector<double>& to_pos_loss,
                                       std::vector<double>& to_neg_loss);


};
#endif
