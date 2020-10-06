#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/pair_optimizer.h"        // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_path = arg_parser.get_str("-train", "", "train graph path");
    std::string save_name = arg_parser.get_str("-save", "deepwalk.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    int walk_times = arg_parser.get_int("-walk_times", 10, "number of being a root in a walk");
    int walk_length = arg_parser.get_int("-walk_length", 40, "length of a walk");
    int window_size = arg_parser.get_int("-window_size", 5, "window size of skipgram");
    int undirected = arg_parser.get_int("-undirected", 1, "whether the graph is undirected");
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
    VCSampler sampler(&file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper vertex_mapper(sampler.vertex_size, dimension); // as vertex
    LookupMapper context_mapper(sampler.vertex_size, dimension); // as context

    // 3. [Optimizer] claim the optimizer
    PairOptimizer optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    unsigned long long total_update_times = (unsigned long long)walk_times*sampler.vertex_size;
    unsigned long long finished_update_times = 0;
    unsigned long long update=0, report_period = 10000;
    double alpha=init_alpha, alpha_min=alpha*0.0001;
    Monitor monitor(total_update_times);

    omp_set_num_threads(worker);
    for (int t=0; t<walk_times; t++)
    {
        #pragma omp parallel for
        for (long index=0; index<sampler.vertex_size; index++)
        {
            long vertex, context;
            std::vector<double> vertex_loss(dimension, 0.0);
            std::vector<double> context_loss(dimension, 0.0);
            std::vector<std::vector<long>> walk = sampler.draw_skipgram(index, walk_length-1, window_size);
            std::vector<long>::iterator it_v = walk[0].begin();
            std::vector<long>::iterator it_c = walk[1].begin();

            for (int i=0; i<walk[0].size(); i++)
            {
                // positive
                vertex = (*it_v);
                context = (*it_c);
                optimizer.feed_loglikelihood_loss(vertex_mapper[vertex], context_mapper[context], 1.0, dimension, vertex_loss, context_loss);
                context_mapper.update_with_l2(context, context_loss, alpha, l2_reg);
                context_loss.assign(dimension, 0.0);

                // negative
                for (int n=0; n<num_negative; n++)
                {
                    context = sampler.draw_a_negative();
                    optimizer.feed_loglikelihood_loss(vertex_mapper[vertex], context_mapper[context], 0.0, dimension, vertex_loss, context_loss);
                    context_mapper.update_with_l2(context, context_loss, alpha, l2_reg);
                    context_loss.assign(dimension, 0.0);
                }
                vertex_mapper.update_with_l2(vertex, vertex_loss, alpha, l2_reg);
                vertex_loss.assign(dimension, 0.0);
                it_v++;
                it_c++;
            }

            // print progress
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
    vertex_mapper.save_to_file(file_graph.index2node, save_name);

    return 0;
}
