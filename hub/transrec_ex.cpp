#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/triplet_optimizer.h"     // optimizer
#include "../src/optimizer/quadruple_optimizer.h"   // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_path = arg_parser.get_str("-train_ui", "", "train graph path");
    std::string meta_path = arg_parser.get_str("-train_im", "", "train graph path");
    std::string save_name = arg_parser.get_str("-save", "transrec.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    int mode = arg_parser.get_int("-mode", 0, "0 for loglikelihood; 1 for bpr");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.025, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.0025, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    unsigned long long total_update_times = (unsigned long long)update_times*1000000;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    // main
    // 0. [FileGraph] read graph
    FileGraph ui_file_graph(train_path, 0);
    FileGraph meta_file_graph(meta_path, 0, ui_file_graph.index2node);

    // 1. [Sampler] determine what sampler to be used
    VCSampler sampler(&ui_file_graph);
    VCSampler meta_sampler(&meta_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(meta_sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    TripletOptimizer optimizer;
    QuadrupleOptimizer bpr_optimizer;

    // 4. building the blocks [BPR]
    std::cout << "Start Training:" << std::endl;
    omp_set_num_threads(worker);
    if (mode == 0)
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        long user, relation, target;
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> relation_loss(dimension, 0.0);
        std::vector<double> target_loss(dimension, 0.0);
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            // user-item-item
            // 4.1 sample positive (user, rel, item_pos) triplet, feed the loss, update
            user = sampler.draw_a_vertex();
            relation = sampler.draw_a_context(user);
            target = sampler.draw_a_context(user);
            optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 1.0, dimension, user_loss, relation_loss, target_loss);
            mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
            mapper.update_with_l2(target, target_loss, alpha, l2_reg);
            relation_loss.assign(dimension, 0.0);
            target_loss.assign(dimension, 0.0);

            // 4.2 sampler negative (user, rel, item_neg) tuple, feed the loss, update
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                target = sampler.draw_a_context_uniformly();
                optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 0.0, dimension, user_loss, relation_loss, target_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target, target_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-item-meta
            // 4.1 sample positive (user, rel, item_pos) triplet, feed the loss, update
            relation = sampler.draw_a_context(user);
            target = sampler.draw_a_context(user);
            target = meta_sampler.draw_a_context(target);
            optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 1.0, dimension, user_loss, relation_loss, target_loss);
            mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
            mapper.update_with_l2(target, target_loss, alpha, l2_reg);
            relation_loss.assign(dimension, 0.0);
            target_loss.assign(dimension, 0.0);

            // 4.2 sampler negative (user, rel, item_neg) tuple, feed the loss, update
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                target = meta_sampler.draw_a_context_uniformly();
                optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 0.0, dimension, user_loss, relation_loss, target_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target, target_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-meta-meta
            // 4.1 sample positive (user, rel, item_pos) triplet, feed the loss, update
            relation = sampler.draw_a_context(user);
            relation = meta_sampler.draw_a_context(relation);
            target = sampler.draw_a_context(user);
            target = meta_sampler.draw_a_context(target);
            optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 1.0, dimension, user_loss, relation_loss, target_loss);
            mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
            mapper.update_with_l2(target, target_loss, alpha, l2_reg);
            relation_loss.assign(dimension, 0.0);
            target_loss.assign(dimension, 0.0);

            // 4.2 sampler negative (user, rel, item_neg) tuple, feed the loss, update
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                relation = meta_sampler.draw_a_context(relation);
                target = meta_sampler.draw_a_context_uniformly();
                optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 0.0, dimension, user_loss, relation_loss, target_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target, target_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-meta-item
            // 4.1 sample positive (user, rel, item_pos) triplet, feed the loss, update
            relation = sampler.draw_a_context(user);
            relation = meta_sampler.draw_a_context(relation);
            target = sampler.draw_a_context(user);
            optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 1.0, dimension, user_loss, relation_loss, target_loss);
            mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
            mapper.update_with_l2(target, target_loss, alpha, l2_reg);
            relation_loss.assign(dimension, 0.0);
            target_loss.assign(dimension, 0.0);

            // 4.2 sampler negative (user, rel, item_neg) tuple, feed the loss, update
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                relation = meta_sampler.draw_a_context(relation);
                target = sampler.draw_a_context_uniformly();
                optimizer.feed_trans_loss(mapper[user], mapper[relation], mapper[target], 0.0, dimension, user_loss, relation_loss, target_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target, target_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // 5. print progress
            update++;
            if (update % report_period == 0) {
                alpha = init_alpha* ( 1.0 - (double)(finished_update_times)/total_update_times );
                if (alpha < alpha_min)
                     alpha = alpha_min;
                finished_update_times += report_period;
                monitor.progress(&finished_update_times);
            }
        }
    }

    if (mode == 1)
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        long user, relation, target_pos, target_neg;
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> relation_loss(dimension, 0.0);
        std::vector<double> target_pos_loss(dimension, 0.0);
        std::vector<double> target_neg_loss(dimension, 0.0);
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            // 4.1 sample the quadruple, feed the loss, update
            user = sampler.draw_a_vertex();

            // user-item-item_pos-item_neg
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                target_pos = sampler.draw_a_context(user);
                target_neg = sampler.draw_a_context_uniformly();
                bpr_optimizer.feed_trans_bpr_loss(mapper[user], mapper[relation], mapper[target_pos], mapper[target_neg],
                                                  dimension, user_loss, relation_loss, target_pos_loss, target_neg_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target_pos, target_pos_loss, alpha, l2_reg);
                mapper.update_with_l2(target_neg, target_neg_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_pos_loss.assign(dimension, 0.0);
                target_neg_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-item-meta_pos-meta_neg
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                target_pos = sampler.draw_a_context(user);
                target_pos = meta_sampler.draw_a_context(target_pos);
                target_neg = meta_sampler.draw_a_context_uniformly();
                bpr_optimizer.feed_trans_bpr_loss(mapper[user], mapper[relation], mapper[target_pos], mapper[target_neg],
                                                  dimension, user_loss, relation_loss, target_pos_loss, target_neg_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target_pos, target_pos_loss, alpha, l2_reg);
                mapper.update_with_l2(target_neg, target_neg_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_pos_loss.assign(dimension, 0.0);
                target_neg_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-meta-item_pos-item_neg
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                relation = meta_sampler.draw_a_context(relation);
                target_pos = sampler.draw_a_context(user);
                target_neg = sampler.draw_a_context_uniformly();
                bpr_optimizer.feed_trans_bpr_loss(mapper[user], mapper[relation], mapper[target_pos], mapper[target_neg],
                                                  dimension, user_loss, relation_loss, target_pos_loss, target_neg_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target_pos, target_pos_loss, alpha, l2_reg);
                mapper.update_with_l2(target_neg, target_neg_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_pos_loss.assign(dimension, 0.0);
                target_neg_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // user-meta-meta_pos-meta_neg
            for (int n=0; n<num_negative; n++)
            {
                relation = sampler.draw_a_context(user);
                relation = meta_sampler.draw_a_context(relation);
                target_pos = sampler.draw_a_context(user);
                target_pos = meta_sampler.draw_a_context(target_pos);
                target_neg = meta_sampler.draw_a_context_uniformly();
                bpr_optimizer.feed_trans_bpr_loss(mapper[user], mapper[relation], mapper[target_pos], mapper[target_neg],
                                                  dimension, user_loss, relation_loss, target_pos_loss, target_neg_loss);
                mapper.update_with_l2(relation, relation_loss, alpha, l2_reg);
                mapper.update_with_l2(target_pos, target_pos_loss, alpha, l2_reg);
                mapper.update_with_l2(target_neg, target_neg_loss, alpha, l2_reg);
                relation_loss.assign(dimension, 0.0);
                target_pos_loss.assign(dimension, 0.0);
                target_neg_loss.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            user_loss.assign(dimension, 0.0);

            // 5. print progress
            update++;
            if (update % report_period == 0) {
                alpha = init_alpha* ( 1.0 - (double)(finished_update_times)/total_update_times );
                if (alpha < alpha_min)
                     alpha = alpha_min;
                finished_update_times += report_period;
                monitor.progress(&finished_update_times);
            }
        }
    }
    monitor.end();
    //mapper.save_to_file(file_graph.index2node, save_name);
    //mapper.save_trans_to_file(&file_graph, save_name);
    //mapper.save_gcn_to_file(&ui_file_graph, ui_file_graph.get_all_from_nodes(), save_name, 0);
    //mapper.save_gcn_to_file(&meta_file_graph, ui_file_graph.get_all_from_nodes(), save_name, 0);
    mapper.save_to_file(&ui_file_graph, ui_file_graph.get_all_nodes(), save_name, 0);
    mapper.save_to_file(&meta_file_graph, ui_file_graph.get_all_to_nodes(), save_name, 1);

    return 0;
}
