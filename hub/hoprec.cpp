#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/triplet_optimizer.h"     // optimizer

int main(int argc, char **argv){

    // arguments
  	ArgParser arg_parser(argc, argv);
    std::string train_path = arg_parser.get_str("-train", "", "train graph path");
    std::string save_name = arg_parser.get_str("-save", "bpr.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.01, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.025, "l2 regularization");
    double margin = arg_parser.get_double("-margin", 1.0, "update margin");
    int num_hop = arg_parser.get_int("-num_hop", 2, "number of hoping steps");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    unsigned long long total_update_times = (unsigned long long)update_times*1000000;
    unsigned long long worker_update_times = total_update_times/worker;
    unsigned long long finished_update_times = 0;
    Monitor monitor(total_update_times);

    // main
    // 0. [FileGraph] read graph
    std::cout << "(UI-Graph)" << std::endl;
    FileGraph file_graph(train_path, 0);
    std::cout << "(Neighbor-Graph)" << std::endl;
    FileGraph neighbor_file_graph(train_path, 1);

    // 1. [Sampler] determine what sampler to be used
    VCSampler sampler(&file_graph);
    VCSampler neighbor_sampler(&neighbor_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    TripletOptimizer optimizer;

    // 4. building the blocks [BPR]
    std::cout << "Start Training:" << std::endl;
    omp_set_num_threads(worker);
    #pragma omp parallel for
    for (int w=0; w<worker; w++)
    {
        long user, item_pos, item_neg;
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss(dimension, 0.0);
        std::vector<long> hopping_list;
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            int real_update_times=0.0;
            // 4.1 sample user with its high-order walking path, feed the loss, update
            user = sampler.draw_a_vertex();
            hopping_list = neighbor_sampler.draw_a_walk(user, num_hop*2);

            // 4.2 sample negative item, feed the loss, update on the hopping path
            for (int hop=1; hop<=num_hop; hop++)
            {
                for (int n=0; n<num_negative; n++)
                {
                    item_pos = hopping_list[(hop-1)*2];
                    item_neg = neighbor_sampler.draw_a_negative();
                    if (optimizer.feed_hoprec_loss(mapper[user], mapper[item_pos], mapper[item_neg], margin/(hop), dimension, user_loss, item_loss)){
                        real_update_times+=1;
                        mapper.update_with_l2(item_pos, item_loss, alpha/hop, l2_reg*(hop));
                        mapper.update_with_l2(item_neg, item_loss, -alpha/hop, -l2_reg);
                        item_loss.assign(dimension, 0.0);
                    }
                }
            }
            if(real_update_times){
                mapper.update_with_l2(user, user_loss, alpha/real_update_times, l2_reg*real_update_times);
                user_loss.assign(dimension, 0.0);
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
    monitor.end();
    mapper.save_to_file(file_graph.index2node, save_name);

    return 0;
}
