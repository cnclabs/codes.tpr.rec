#include "triplet_optimizer.h"

TripletOptimizer::TripletOptimizer() {
    // pre-compute sigmoid func
    this->cached_sigmoid.resize(SIGMOID_TABLE_SIZE);
    for (int i = 0; i != SIGMOID_TABLE_SIZE + 1; i++)
    {
        double x = i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        this->cached_sigmoid[i] = 1.0 / (1.0 + exp(-x));
    }
}

double TripletOptimizer::fast_sigmoid(double value) {
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

int TripletOptimizer::feed_margin_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& to_embedding_pos, std::vector<double>& to_embedding_neg, double margin, int dimension, std::vector<double>& from_loss, std::vector<double>& to_loss_pos, std::vector<double>& to_loss_neg) {

    std::vector<double> diff_to_embedding;
    diff_to_embedding.resize(dimension, 0.0);

    double gradient, prediction=-margin;

    for (int d=0; d<dimension;d++)
    {
        diff_to_embedding[d] = to_embedding_pos[d] - to_embedding_neg[d];
        prediction += from_embedding[d] * ( diff_to_embedding[d] );
    }

    gradient = this->fast_sigmoid(0.0-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * diff_to_embedding[d];
        to_loss_pos[d] += gradient * from_embedding[d];
        to_loss_neg[d] -= gradient * from_embedding[d];
    }
    return 1;
}

void TripletOptimizer::feed_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& to_embedding_pos, std::vector<double>& to_embedding_neg, int dimension, std::vector<double>& from_loss, std::vector<double>& to_loss) {

    std::vector<double> diff_to_embedding;
    diff_to_embedding.resize(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; d++)
    {
        diff_to_embedding[d] = to_embedding_pos[d] - to_embedding_neg[d];
        prediction += from_embedding[d] * ( diff_to_embedding[d] );
    }

    gradient = this->fast_sigmoid(0.0-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * diff_to_embedding[d];
        to_loss[d] += gradient * from_embedding[d];
    }
}

int TripletOptimizer::feed_hoprec_loss(std::vector<double>& from_embedding, std::vector<double>& to_embedding_pos, std::vector<double>& to_embedding_neg, double margin, int dimension, std::vector<double>& from_loss, std::vector<double>& to_loss) {

    std::vector<double> diff_to_embedding;
    diff_to_embedding.resize(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension;d++)
    {
        diff_to_embedding[d] = to_embedding_pos[d] - to_embedding_neg[d];
        prediction += from_embedding[d] * ( diff_to_embedding[d] );
    }

    if(prediction>margin) return 0;

    gradient = this->fast_sigmoid(0.0-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * diff_to_embedding[d];
        to_loss[d] += gradient * from_embedding[d];
    }
    return 1;
}

void TripletOptimizer::feed_trans_loss(std::vector<double>& from_embedding, std::vector<double>& relation_embedding, std::vector<double>& to_embedding, double label, int dimension, std::vector<double>& from_loss, std::vector<double>& relation_loss, std::vector<double>& to_loss) {

    std::vector<double> fused_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        fused_embedding[d] = from_embedding[d] + relation_embedding[d];
        prediction += fused_embedding[d] * to_embedding[d];
    }

    gradient = label - this->fast_sigmoid(prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * to_embedding[d];
        relation_loss[d] += gradient * to_embedding[d];
        to_loss[d] += gradient * fused_embedding[d];
    }
}

double TripletOptimizer::skew_opt(double prediction, double location, double scale) {

    double gradient, skew_prediction = 0;

    skew_prediction = (prediction-location)/scale;

    if(skew_prediction>2.0) return 0;
    if(skew_prediction<2.0) skew_prediction = -2.0;

    gradient = this->fast_sigmoid(-skew_prediction*skew_prediction*skew_prediction)* skew_prediction*skew_prediction/scale;

    return gradient;
}

int TripletOptimizer::feed_skew_opt_loss(std::vector<double>& from_embedding, std::vector<double>& to_embedding_pos, std::vector<double>& to_embedding_neg, double location, double scale, int dimension, std::vector<double>& from_loss, std::vector<double>& to_loss) {

    std::vector<double> diff_to_embedding;
    diff_to_embedding.resize(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension;d++)
    {
        diff_to_embedding[d] = to_embedding_pos[d] - to_embedding_neg[d];
        prediction += from_embedding[d] * diff_to_embedding[d];
    }

    gradient = this->skew_opt(prediction, location, scale);

    if(gradient==0) return 0;

    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * diff_to_embedding[d];
        to_loss[d] += gradient * from_embedding[d];
    }
    return 1;
}
