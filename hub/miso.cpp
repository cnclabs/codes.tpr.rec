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
    std::string train_ui_path = arg_parser.get_str("-train_ui", "", "input user-item graph path");
    std::string train_iw_path = arg_parser.get_str("-train_iw", "", "input item-word graph path");
    std::string save_name = arg_parser.get_str("-save", "cse.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.1, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.01, "l2 regularization");
    double l2_reg2 = arg_parser.get_double("-l2_reg2", 0.01, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // main
    // 0. [FileGraph] read graph
    std::cout << "(UI-Graph)" << std::endl;
    FileGraph ui_file_graph(train_ui_path, 0);
    std::cout << "(IW-Graph)" << std::endl;
    FileGraph iw_file_graph(train_iw_path, 0, ui_file_graph.index2node);

    // 1. [Sampler] determine what sampler to be used
    VCSampler ui_sampler(&ui_file_graph);
    VCSampler iw_sampler(&iw_file_graph);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper(iw_sampler.vertex_size, dimension);

    // 3. [Optimizer] claim the optimizer
    TripletOptimizer optimizer;

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
        long user, item_given, item_pos, item_neg;
        std::vector<double> user_embed(dimension, 0.0);
        std::vector<double> item_embed_pos(dimension, 0.0);
        std::vector<double> item_embed_neg(dimension, 0.0);
        std::vector<double> user_loss(dimension, 0.0);
        std::vector<double> item_loss_pos(dimension, 0.0);
        std::vector<double> item_loss_neg(dimension, 0.0);
        std::vector<long> user2items, item2words_pos, item2words_neg;
        unsigned long long update=0, report_period = 10000;
        double alpha=init_alpha, alpha_min=alpha*0.0001;
        int trial;

        while (update < worker_update_times)
        {
            user = ui_sampler.draw_a_vertex();
            item_given = ui_sampler.draw_a_context(user);
            user2items.clear();
            // [user, item, item, ...]
            user2items.push_back(user);
            user2items.push_back(item_given);
            //ui_sampler.feed_sampled_contexts(user, num_item, user2items); // user-items
            user_embed = mapper.textgcn_embedding(user2items);

            for (int b=0; b<5; b++)
            {
                item2words_pos.clear();
                item_pos = ui_sampler.draw_a_context(user);
                // [item, word, word, ...]
                item2words_pos.push_back(item_pos);
                iw_sampler.feed_sampled_contexts(item_pos, 1, item2words_pos); // item-words
                item_embed_pos = mapper.textgcn_embedding(item2words_pos);

                item2words_neg.clear();
                item_neg = ui_sampler.draw_a_context_uniformly();
                // [user, word, word, ...]
                item2words_neg.push_back(item_neg);
                iw_sampler.feed_sampled_contexts(item_neg, 1, item2words_neg); // item-words
                item_embed_neg = mapper.textgcn_embedding(item2words_neg);

                optimizer.feed_margin_bpr_loss(user_embed, item_embed_pos, item_embed_neg, 8.0, dimension, user_loss, item_loss_pos, item_loss_neg);
                for (auto it=item2words_pos.begin(); it!=item2words_pos.end(); it++)
                    mapper.update_with_l2(*it, item_loss_pos, alpha, l2_reg);
                for (auto it=item2words_neg.begin(); it!=item2words_neg.end(); it++)
                    mapper.update_with_l2(*it, item_loss_neg, alpha, l2_reg);
                item_loss_pos.assign(dimension, 0.0);
                item_loss_neg.assign(dimension, 0.0);
            }
            mapper.update_with_l2(user, user_loss, alpha, l2_reg);
            mapper.update_with_l2(item_given, user_loss, alpha, l2_reg);
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
    // get union of item nodes
    std::set<long> item_set;
    for (auto node: ui_file_graph.get_all_to_nodes())
        item_set.insert(node);
    for (auto node: iw_file_graph.get_all_from_nodes())
        item_set.insert(node);
    std::vector<long> all_item_nodes(item_set.begin(), item_set.end());
    // save embeddings
    mapper.save_gcn_to_file(&ui_file_graph, ui_file_graph.get_all_from_nodes(), save_name, 0);
    mapper.save_gcn_to_file(&iw_file_graph, all_item_nodes, save_name, 1);
    mapper.save_to_file(&iw_file_graph, iw_file_graph.get_all_to_nodes(), save_name, 1);

    return 0;
}
