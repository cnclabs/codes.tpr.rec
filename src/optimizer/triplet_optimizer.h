#ifndef TRIPLET_OPTIMIZER_H
#define TRIPLET_OPTIMIZER_H
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

#define SIGMOID_TABLE_SIZE 1000
#define MAX_SIGMOID 8.0

class TripletOptimizer {
    private:
        void init_sigmoid();

    public:
        // constructor
        TripletOptimizer();

        // variables
        std::vector<double> cached_sigmoid;

        // functions
        double fast_sigmoid(double value);
        double skew_opt(double prediction, double location, double scale);

        // loss
        int feed_margin_bpr_loss(std::vector<double>& from_embedding,
                                 std::vector<double>& to_embedding_pos,
                                 std::vector<double>& to_embedding_neg,
                                 double margin,
                                 int dimension,
                                 std::vector<double>& from_loss,
                                 std::vector<double>& to_loss_pos,
                                 std::vector<double>& to_loss_neg);
        void feed_bpr_loss(std::vector<double>& from_embedding,
                           std::vector<double>& to_embedding_pos,
                           std::vector<double>& to_embedding_neg,
                           int dimension,
                           std::vector<double>& from_loss,
                           std::vector<double>& to_loss);
        int feed_hoprec_loss(std::vector<double>& from_embedding,
                             std::vector<double>& to_embedding_pos,
                             std::vector<double>& to_embedding_neg,
                             double margin,
                             int dimension,
                             std::vector<double>& from_loss,
                             std::vector<double>& to_loss);
        void feed_trans_loss(std::vector<double>& from_embedding,
                             std::vector<double>& relation_embedding,
                             std::vector<double>& to_embedding,
                             double label,
                             int dimension,
                             std::vector<double>& from_loss,
                             std::vector<double>& relation_loss,
                             std::vector<double>& to_loss);
        int feed_skew_opt_loss(std::vector<double>& from_embedding,
                             std::vector<double>& to_embedding_pos,
                             std::vector<double>& to_embedding_neg,
                             double location,
                             double scale,
                             int dimension,
                             std::vector<double>& from_loss,
                             std::vector<double>& to_loss);


};
#endif
