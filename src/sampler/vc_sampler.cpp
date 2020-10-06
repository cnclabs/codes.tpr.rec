#include "vc_sampler.h"

VCSampler::VCSampler(FileGraph* file_graph) {
    /* See Figure 3. in https://arxiv.org/abs/1711.00227
     */
    this->vertex_size = file_graph->index2node.size();
    this->context_size = file_graph->index_graph.size();

    std::cout << "Build VC Sampler:" << std::endl;
    long from_index, to_index, to_offset;
    double weight;
    std::vector<double> vertex_distribution, context_distribution, neg_distribution;
    std::vector<double> vertex_uniform_distribution, context_uniform_distribution;
    vertex_distribution.resize(this->vertex_size, 0.0);
    vertex_uniform_distribution.resize(this->vertex_size, 0.0);
    context_uniform_distribution.resize(this->vertex_size, 0.0);
    neg_distribution.resize(this->vertex_size, 0.0);

    std::cout << "\tBuild Alias Methods" << std::endl;
    for (int from_index=0; from_index<this->vertex_size; from_index++)
    {
        context_distribution.clear();
        for (auto it: file_graph->index_graph[from_index])
        {
            to_index = it.first;
            weight = it.second;
            vertex_distribution[from_index] += weight;
            context_distribution.push_back(weight);
            neg_distribution[to_index] += weight;
            vertex_uniform_distribution[from_index] = 1.0;
            context_uniform_distribution[to_index] = 1.0;
            this->contexts.push_back(to_index);
        }
        this->context_sampler.append(context_distribution, 1.0);
    }
    this->vertex_sampler.append(vertex_distribution, 1.0);
    this->vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0);
    this->context_uniform_sampler.append(context_uniform_distribution, 1.0);
    this->negative_sampler.append(neg_distribution, 0.75);
    std::cout << "\tDone" << std::endl;
}

long VCSampler::draw_a_vertex() {
    return this->vertex_sampler.draw();
}

long VCSampler::draw_a_context(long vertex_index) {
    return this->contexts[this->context_sampler.draw(vertex_index)];
}

std::vector<long> VCSampler::get_neighbors(long vertex_index) {
    std::vector<long> neighbors;
    long offset = this->context_sampler.get_offset(vertex_index);
    long branch = this->context_sampler.get_branch(vertex_index);
    for (int i=0; i<branch; i++)
    {
        neighbors.push_back(this->contexts[offset+i]);
    }
    return neighbors;
}

void VCSampler::feed_sampled_contexts(long vertex_index, int num_sample, std::vector<long>& feed_me) {
    for (int i=0; i<num_sample; i++)
    {
        long sample = this->draw_a_context_safely(vertex_index);
        if (sample==-1) return;
        feed_me.push_back(sample);
    }
}

void VCSampler::feed_all_neighbors(long vertex_index, std::vector<long>& feed_me) {
    long offset = this->context_sampler.get_offset(vertex_index);
    long branch = this->context_sampler.get_branch(vertex_index);
    for (int i=0; i<branch; i++)
    {
        feed_me.push_back(this->contexts[offset+i]);
    }
}

long VCSampler::draw_a_context_safely(long vertex_index) {
    long context_index = this->context_sampler.draw_safely(vertex_index);
    if (context_index != -1)
        return this->contexts[context_index];
    return -1;
}

long VCSampler::draw_a_negative() {
    return this->negative_sampler.draw();
}

long VCSampler::draw_a_vertex_uniformly() {
    return this->vertex_uniform_sampler.draw();
}

long VCSampler::draw_a_context_uniformly() {
    return this->context_uniform_sampler.draw();
}

std::vector<long> VCSampler::draw_a_walk(int walk_steps) {
    std::vector<long> walk;
    long node;
    node = this->draw_a_vertex();
    walk.push_back(node);
    for (int w=1; w<walk_steps; w++)
    {
        node = this->draw_a_context(node);
        walk.push_back(node);
    }
    return walk;
}

std::vector<long> VCSampler::draw_a_walk(long node, int walk_steps) {
    std::vector<long> walk;
    for (int w=0; w<walk_steps; w++)
    {
        node = this->draw_a_context_safely(node);
        if (node==-1)
            return walk;
        walk.push_back(node);
    }
    return walk;
}

std::vector<long> VCSampler::draw_a_jump_walk(long node, double jump_prob) {
    std::vector<long> walk;
    while (1)
    {
        node = this->draw_a_context_safely(node);
        if (node==-1)
            return walk;
        walk.push_back(node);
        if (random_range(0.0, 1.0) < jump_prob) break;
    }
    return walk;
}

std::vector<std::vector<long>> VCSampler::draw_skipgram(long node, int walk_length, int window_size) {
    std::vector<std::vector<long>> pairs;
    std::vector<long> vertexes, contexts;
    std::vector<long> walk = this->draw_a_walk(node, walk_length);
    int left, right, reduce;
    walk_length = walk.size(); // walk might be short because of no connection
    for (int i=0; i<walk_length; i++)
    {
        reduce = random_range(0, window_size) + 1;
        left = i-reduce;
        if (left < 0) left = 0;
        right = i+reduce;
        if (right > walk_length) right = walk_length;

        for (int j=left; j<=right; j++)
        {
            if (i==j) continue;
            vertexes.push_back(walk[i]);
            contexts.push_back(walk[j]);
        }
    }
    pairs.push_back(vertexes);
    pairs.push_back(contexts);
    return pairs;
}

std::vector<std::vector<long>> VCSampler::draw_scaledskipgram(long node, int walk_length, int window_min, int window_max) {
    std::vector<std::vector<long>> pairs;
    std::vector<long> vertexes, contexts;
    std::vector<long> walk = this->draw_a_walk(node, walk_length);
    int left, right, reduce;
    walk_length = walk.size(); // walk might be short because of no connection
    for (int i=0; i<walk_length; i++)
    {
        left = i-window_max;
        if (left < 0) left = 0;
        right = i-window_min;
        if (right < 0) right = 0;

        for (int j=left; j<=right; j++)
        {
            if (i==j) continue;
            vertexes.push_back(walk[i]);
            contexts.push_back(walk[j]);
        }

        left = i+window_min;
        if (left > walk_length) left = walk_length;
        right = i+window_max;
        if (right > walk_length) right = walk_length;

        for (int j=left; j<=right; j++)
        {
            if (i==j) continue;
            vertexes.push_back(walk[i]);
            contexts.push_back(walk[j]);
        }
    }
    pairs.push_back(vertexes);
    pairs.push_back(contexts);
    return pairs;
}

