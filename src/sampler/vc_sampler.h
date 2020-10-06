#ifndef VC_SAMPLER_H
#define VC_SAMPLER_H

#include <cmath>
#include <unordered_map>
#include <vector>
#include "../util/file_graph.h"
#include "../util/random.h"
#include "alias_methods.h"

class VCSampler {
    /* VCSampler performs vertex-context-style sampling
     */
    public:
        VCSampler(FileGraph*);

        // variables
        long vertex_size=0, context_size=0;
        AliasMethods vertex_sampler, context_sampler, negative_sampler;
        AliasMethods vertex_uniform_sampler, context_uniform_sampler;
        std::vector<long> contexts; // context ref.
        //std::unordered_map<long, std::vector<long>> adjacency; // context ref.

        // functions
        long draw_a_vertex();
        long draw_a_vertex_uniformly();
        long draw_a_context(long vertex);
        long draw_a_context_uniformly();
        long draw_a_context_safely(long vertex);
        long draw_a_negative();
        std::vector<long> draw_a_walk(int walk_steps);
        std::vector<long> draw_a_walk(long node, int walk_steps);
        std::vector<long> draw_a_jump_walk(long node, double jump_prob);
        std::vector<long> get_neighbors(long node);
        void feed_sampled_contexts(long vertex_index, int num_sample, std::vector<long>& feed_me);
        void feed_all_neighbors(long vertex_index, std::vector<long>& feed_me);
        std::vector<std::vector<long>> draw_skipgram(long node, int walk_length, int window_size);
        std::vector<std::vector<long>> draw_scaledskipgram(long node, int walk_length, int window_min, int window_max);
};
#endif
