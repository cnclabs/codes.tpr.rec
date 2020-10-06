#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "../src/util/util.h"                       // arguments
#include "../src/util/random.h"                     // randomizer
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/edge_sampler.h"            // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/pair_optimizer.h"        // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_path = arg_parser.get_str("-train", "", "train graph path");
    std::string save_name = arg_parser.get_str("-save", "implicit_mf.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.025, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // main
    // 0. [FileGraph] read graph
    FileGraph file_graph(train_path, 0);

    // 1. [Sampler] determine what sampler to be used
    EdgeSampler sampler(&file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(sampler.node_size, dimension);

    // 3. [Optimizer] claim the optimizer
    PairOptimizer optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    unsigned long long total_update_times = (unsigned long long)update_times*1000000;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    omp_set_num_threads(worker);
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        std::vector<long> edge;
        long user, item;
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss(dimension, 0.0);
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            // 4.1 sample positive (user, item) pair, feed the loss, update
            sampler.feed_an_edge(&user, &item);
            optimizer.feed_dotproduct_loss(mapper[user], mapper[item], 1.0, dimension, user_loss, item_loss);
            mapper.update_with_l2(item, item_loss, alpha, l2_reg);
            item_loss.assign(dimension, 0.0);

            // 4.2 sampler negative (user, item) pair, feed the loss, update
            for (int n=0; n<num_negative; n++)
            {
                item = sampler.draw_a_negative();
                optimizer.feed_dotproduct_loss(mapper[user], mapper[item], -1.0, dimension, user_loss, item_loss);
                mapper.update_with_l2(item, item_loss, alpha, l2_reg);
                item_loss.assign(dimension, 0.0);
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
    mapper.save_to_file(file_graph.index2node, save_name);

    return 0;
}
