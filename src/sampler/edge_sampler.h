#ifndef EDGE_SAMPLER_H
#define EDGE_SAMPLER_H

#include <cmath>
#include <unordered_map>
#include <vector>
#include "../util/file_graph.h"
#include "../util/random.h"
#include "alias_methods.h"

class EdgeSampler {
    /* EdgeSampler performs edge-style sampling
     */
    public:
        EdgeSampler(FileGraph*);

        // variables
        long node_size=0, edge_size=0, context_size=0;
        AliasMethods vertex_uniform_sampler, context_uniform_sampler;
        AliasMethods edge_sampler, node_sampler, negative_sampler;
        std::vector<long> vertexes, contexts, labels;
        std::vector<long> unique_contexts;
        std::vector<long> offset, branch;

        // functions
        std::vector<long> draw_an_edge();
        void feed_an_edge(long* from_node, long* to_node);
        long draw_a_vertex();
        long draw_a_context();
        long draw_a_node();
        long draw_a_node_uniformly();
        long draw_a_negative();
        long draw_a_vertex_uniformly();
        long draw_a_context_uniformly();
        std::vector<long> get_neighbors(long node);
};
#endif
