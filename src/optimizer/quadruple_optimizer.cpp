#include "quadruple_optimizer.h"

QuadrupleOptimizer::QuadrupleOptimizer() {
    // pre-compute sigmoid func
    this->cached_sigmoid.resize(SIGMOID_TABLE_SIZE);
    for (int i = 0; i != SIGMOID_TABLE_SIZE + 1; i++)
    {
        double x = i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        this->cached_sigmoid[i] = 1.0 / (1.0 + exp(-x));
    }
}

double QuadrupleOptimizer::fast_sigmoid(double value) {
    if (value < -MAX_SIGMOID)
    {
        return 0.0;
    }
    else if (value > MAX_SIGMOID)
    {
        return 1.0;
    }
    else
    {
        return this->cached_sigmoid[ int((value + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2) ];
    }
}

void QuadrupleOptimizer::feed_trans_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& relation_embedding,
                                             std::vector<double>& to_pos_embedding, std::vector<double>& to_neg_embedding,
                                             int dimension,
                                             std::vector<double>& from_loss, std::vector<double>& relation_loss,
                                             std::vector<double>& to_pos_loss, std::vector<double>& to_neg_loss) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] = from_embedding[d] + relation_embedding[d];
        target_embedding[d] = to_pos_embedding[d] - to_neg_embedding[d];
        prediction += source_embedding[d] * target_embedding[d];
    }

    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * target_embedding[d];
        relation_loss[d] += gradient * target_embedding[d];
        to_pos_loss[d] += gradient * source_embedding[d];
        to_neg_loss[d] -= gradient * source_embedding[d];
    }
}

int QuadrupleOptimizer::feed_trans_margin_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& relation_embedding,
                                                   std::vector<double>& to_pos_embedding, std::vector<double>& to_neg_embedding,
                                                   double margin, int dimension,
                                                   std::vector<double>& from_loss, std::vector<double>& relation_loss,
                                                   std::vector<double>& to_pos_loss, std::vector<double>& to_neg_loss) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] = from_embedding[d] + relation_embedding[d];
        target_embedding[d] = to_pos_embedding[d] - to_neg_embedding[d];
        prediction += source_embedding[d] * target_embedding[d];
    }
    if (prediction > margin)
        return 0;

    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * target_embedding[d];
        relation_loss[d] += gradient * target_embedding[d];
        to_pos_loss[d] += gradient * source_embedding[d];
        to_neg_loss[d] -= gradient * source_embedding[d];
    }
    return 1;
}
