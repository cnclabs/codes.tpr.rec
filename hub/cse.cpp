#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/pair_optimizer.h"        // optimizer
#include "../src/optimizer/triplet_optimizer.h"     // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_path = arg_parser.get_str("-train", "", "train graph path");
    std::string save_name = arg_parser.get_str("-save", "cse.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    int walk_step = arg_parser.get_int("-walk_step", 2, "walk steps for retriving neighbors");
    int mode = arg_parser.get_int("-mode", 1, "0 for 0-1-rating; 1 for ranking");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.025, "init learning rate");
    double margin = arg_parser.get_double("-margin", 1.0, "margin for ranking");
    double lambda = arg_parser.get_double("-lambda", 0.05, "learning ratio of neighborhood modeling");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // main
    // 0. [FileGraph] read graph
    std::cout << "(UI-Graph)" << std::endl;
    FileGraph ui_file_graph(train_path, 0);
    std::cout << "(Neighbor-Graph)" << std::endl;
    FileGraph neighbor_file_graph(train_path, 1);

    // 1. [Sampler] determine what sampler to be used
    VCSampler ui_sampler(&ui_file_graph);
    VCSampler neighbor_sampler(&neighbor_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper ui_mapper(ui_sampler.vertex_size, dimension);  // for ui
    LookupMapper u_mapper(ui_sampler.vertex_size, dimension);   // for u's neighbors
    LookupMapper i_mapper(ui_sampler.vertex_size, dimension);   // for i's neighbors

    // 3. [Optimizer] claim the optimizer
    PairOptimizer optimizer;
    TripletOptimizer bpr_optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    unsigned long long total_update_times = (unsigned long long)update_times*1000000;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    omp_set_num_threads(worker);
    if (mode== 1)
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        int step;
        long user, item, item_neg;
        std::vector<double> ui_loss(dimension, 0.0);
        std::vector<double> u_loss(dimension, 0.0);
        std::vector<double> i_loss(dimension, 0.0);
        std::vector<double> i_loss_pos(dimension, 0.0);
        std::vector<double> i_loss_neg(dimension, 0.0);
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            user = ui_sampler.draw_a_vertex();
            item = ui_sampler.draw_a_context(user);

            // item-neighbors
            for (auto neighbor: neighbor_sampler.draw_a_walk(item, walk_step))
            {
                // positive
                optimizer.feed_loglikelihood_loss(ui_mapper[item], i_mapper[neighbor], 1.0, dimension, ui_loss, i_loss);
                i_mapper.update_with_l2(neighbor, i_loss, alpha*lambda, l2_reg);
                i_loss.assign(dimension, 0.0);

                // negative
                for (int n=0; n<num_negative; n++)
                {
                    neighbor = neighbor_sampler.draw_a_negative();
                    optimizer.feed_loglikelihood_loss(ui_mapper[item], i_mapper[neighbor], 0.0, dimension, ui_loss, i_loss);
                    i_mapper.update_with_l2(neighbor, i_loss, alpha*lambda, l2_reg);
                    i_loss.assign(dimension, 0.0);
                }
                ui_mapper.update_with_l2(item, ui_loss, alpha*lambda, l2_reg);
                ui_loss.assign(dimension, 0.0);
            }

            // user-neighbors
            for (auto neighbor: neighbor_sampler.draw_a_walk(user, walk_step))
            {
                // positive
                optimizer.feed_loglikelihood_loss(ui_mapper[user], u_mapper[neighbor], 1.0, dimension, ui_loss, u_loss);
                u_mapper.update_with_l2(neighbor, u_loss, alpha*lambda, l2_reg);
                u_loss.assign(dimension, 0.0);

                // negative
                for (int n=0; n<num_negative; n++)
                {
                    neighbor = neighbor_sampler.draw_a_negative();
                    optimizer.feed_loglikelihood_loss(ui_mapper[user], u_mapper[neighbor], 0.0, dimension, ui_loss, u_loss);
                    u_mapper.update_with_l2(neighbor, u_loss, alpha*lambda, l2_reg);
                    u_loss.assign(dimension, 0.0);
                }
                ui_mapper.update_with_l2(user, ui_loss, alpha*lambda, l2_reg);
                ui_loss.assign(dimension, 0.0);
            }

            // user-item_pos-item_neg
            for (int s=0; s<16; s++)
            {
                item_neg = ui_sampler.draw_a_context_uniformly();
                if (bpr_optimizer.feed_margin_bpr_loss(ui_mapper[user], ui_mapper[item], ui_mapper[item_neg], margin, dimension, u_loss, i_loss_pos, i_loss_neg))
                {
                    ui_mapper.update_with_l2(item, i_loss_pos, alpha, l2_reg);
                    ui_mapper.update_with_l2(item_neg, i_loss_neg, alpha, l2_reg);
                    i_loss_pos.assign(dimension, 0.0);
                    i_loss_neg.assign(dimension, 0.0);
                    ui_mapper.update_with_l2(user, u_loss, alpha, l2_reg);
                    u_loss.assign(dimension, 0.0);
                    break;
                }
            }

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

    if (mode== 0)
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        int step;
        long user, item;
        std::vector<double> ui_loss(dimension, 0.0);
        std::vector<double> u_loss(dimension, 0.0);
        std::vector<double> i_loss(dimension, 0.0);
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            user = ui_sampler.draw_a_vertex();
            item = ui_sampler.draw_a_context(user);

            // item-neighbors
            for (auto neighbor: neighbor_sampler.draw_a_walk(item, walk_step))
            {
                // positive
                optimizer.feed_loglikelihood_loss(ui_mapper[item], i_mapper[neighbor], 1.0, dimension, ui_loss, i_loss);
                i_mapper.update_with_l2(neighbor, i_loss, alpha*lambda, l2_reg);
                i_loss.assign(dimension, 0.0);

                // negative
                for (int n=0; n<num_negative; n++)
                {
                    neighbor = neighbor_sampler.draw_a_negative();
                    optimizer.feed_loglikelihood_loss(ui_mapper[item], i_mapper[neighbor], 0.0, dimension, ui_loss, i_loss);
                    i_mapper.update_with_l2(neighbor, i_loss, alpha*lambda, l2_reg);
                    i_loss.assign(dimension, 0.0);
                }
                ui_mapper.update_with_l2(item, ui_loss, alpha*lambda, l2_reg);
                ui_loss.assign(dimension, 0.0);
            }

            // user-neighbors
            for (auto neighbor: neighbor_sampler.draw_a_walk(user, walk_step))
            {
                // positive
                optimizer.feed_loglikelihood_loss(ui_mapper[user], u_mapper[neighbor], 1.0, dimension, ui_loss, u_loss);
                u_mapper.update_with_l2(neighbor, u_loss, alpha*lambda, l2_reg);
                u_loss.assign(dimension, 0.0);

                // negative
                for (int n=0; n<num_negative; n++)
                {
                    neighbor = neighbor_sampler.draw_a_negative();
                    optimizer.feed_loglikelihood_loss(ui_mapper[user], u_mapper[neighbor], 0.0, dimension, ui_loss, u_loss);
                    u_mapper.update_with_l2(neighbor, u_loss, alpha*lambda, l2_reg);
                    u_loss.assign(dimension, 0.0);
                }
                ui_mapper.update_with_l2(user, ui_loss, alpha*lambda, l2_reg);
                ui_loss.assign(dimension, 0.0);
            }

            // userr-items
            // positive
            optimizer.feed_loglikelihood_loss(ui_mapper[user], ui_mapper[item], 1.0, dimension, u_loss, i_loss);
            ui_mapper.update_with_l2(item, i_loss, alpha, l2_reg);
            i_loss.assign(dimension, 0.0);
            // negative
            for (int n=0; n<num_negative; n++)
            {
                item = ui_sampler.draw_a_context_uniformly();
                optimizer.feed_loglikelihood_loss(ui_mapper[user], ui_mapper[item], 0.0, dimension, u_loss, i_loss);
                ui_mapper.update_with_l2(item, i_loss, alpha, l2_reg);
                i_loss.assign(dimension, 0.0);
            }
            ui_mapper.update_with_l2(user, u_loss, alpha, l2_reg);
            u_loss.assign(dimension, 0.0);

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
    ui_mapper.save_to_file(ui_file_graph.index2node, save_name);

    return 0;
}
