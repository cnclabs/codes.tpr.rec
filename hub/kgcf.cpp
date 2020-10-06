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
    std::string train_ui_path = arg_parser.get_str("-train_ui", "", "input user-item graph path");
    std::string train_ik_path = arg_parser.get_str("-train_ik", "", "input item-knowledge graph path");
    std::string save_name = arg_parser.get_str("-save", "kgcf.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    int use_kg = arg_parser.get_int("-use_kg", 1, "use kg or not");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.1, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // main
    // 0. [FileGraph] read graph
    std::cout << "(UI-Graph)" << std::endl;
    FileGraph ui_file_graph(train_ui_path, 0); // for sample a user-item pair
    FileGraph uiui_file_graph(train_ui_path, 1, ui_file_graph.index2node); // for user-item random walk
    std::cout << "(IK-Graph)" << std::endl;
    FileGraph ik_file_graph(train_ik_path, 0, ui_file_graph.index2node); // for sample item-knowledge

    // 1. [Sampler] determine what sampler to be used
    VCSampler ui_sampler(&ui_file_graph);
    VCSampler uiui_sampler(&uiui_file_graph);
    VCSampler ik_sampler(&ik_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(ik_sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    PairOptimizer kg_optimizer;
    TripletOptimizer bpr_optimizer;

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
        int step;
        long user, kg_pos, kg_neg, _user, _item, item_pos, item_neg, fail;
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss(dimension, 0.0);
        std::vector<double> kg_loss(dimension, 0.0);
        std::vector<long> user2items, item2kg_pos, item2kg_neg;
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;

        while (update < worker_update_times)
        {
            user = ui_sampler.draw_a_vertex();

            for (int n=0; n<num_negative; n++)
            {
                item_pos = ui_sampler.draw_a_context(user);
                item_neg = ui_sampler.draw_a_context_uniformly();

                if (use_kg)
                {
                    // positive item-kg (2nd-order modeling)
                    kg_pos = -1;
                    fail = 0;
                    while (kg_pos==-1)
                    {
                        _item = uiui_sampler.draw_a_context(user);
                        kg_pos = ik_sampler.draw_a_context_safely(_item);
                        if (fail>5)
                        {
                            kg_pos = ik_sampler.draw_a_context_uniformly();
                        }
                        fail++;
                    }
                    kg_neg = ik_sampler.draw_a_context_uniformly();
                    bpr_optimizer.feed_bpr_loss(mapper[item_pos], mapper[kg_pos], mapper[kg_neg], dimension, item_loss, kg_loss);
                    mapper.update_with_l2(item_pos, item_loss, alpha*0.15, l2_reg); // try here
                    mapper.update_with_l2(kg_pos, kg_loss, alpha*0.15, l2_reg);
                    mapper.update_with_l2(kg_neg, kg_loss, -alpha*0.15, -l2_reg);
                    kg_loss.assign(dimension, 0.0);
                    item_loss.assign(dimension, 0.0);
                }

                // BPR modeling
                bpr_optimizer.feed_bpr_loss(mapper[user], mapper[item_pos], mapper[item_neg], dimension, user_loss, item_loss);
                mapper.update_with_l2(user, user_loss, alpha, l2_reg);
                mapper.update_with_l2(item_pos, item_loss, alpha, l2_reg);
                mapper.update_with_l2(item_neg, item_loss, -alpha, -l2_reg);
                user_loss.assign(dimension, 0.0);
                item_loss.assign(dimension, 0.0);
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
    mapper.save_to_file(&ui_file_graph, ui_file_graph.get_all_nodes(), save_name, 0); // new file

    return 0;
}
