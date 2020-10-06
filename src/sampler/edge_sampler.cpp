#include "edge_sampler.h"

EdgeSampler::EdgeSampler(FileGraph* file_graph) {
    /* See Figure 3. in https://arxiv.org/abs/1711.00227
     */
    this->node_size = file_graph->index2node.size();
    this->edge_size = file_graph->edge_size;

    std::cout << "Build Edge Sampler:" << std::endl;
    long from_index, to_index, to_offset;
    long num_unique_contexts;
    double weight;
    std::vector<double> node_distribution, edge_distribution, neg_distribution;
    std::vector<double> vertex_uniform_distribution, context_uniform_distribution;
    vertex_uniform_distribution.resize(this->node_size, 0.0); // uniform
    context_uniform_distribution.resize(this->node_size, 0.0); // uniform
    node_distribution.resize(this->node_size, 0.0); // indegree + outdegree
    neg_distribution.resize(this->node_size, 0.0); // indegree
    edge_distribution.reserve(this->edge_size); // degree

    std::cout << "\tBuild Alias Methods" << std::endl;
    for (int from_index=0; from_index<this->node_size; from_index++)
    {
        this->offset.push_back(edge_distribution.size());
        this->branch.push_back(file_graph->index_graph[from_index].size());
        for (auto it: file_graph->index_graph[from_index])
        {
            to_index = it.first;
            weight = it.second;
            if (context_uniform_distribution[to_index] != 1.0)
            {
                this->unique_contexts.push_back(to_index);
                this->context_size++;
            }
            vertex_uniform_distribution[from_index] = 1.0;
            context_uniform_distribution[to_index] = 1.0;
            node_distribution[from_index] += weight;
            node_distribution[to_index] += weight;
            neg_distribution[to_index] += weight;
            edge_distribution.push_back(weight);
            this->vertexes.push_back(from_index);
            this->contexts.push_back(to_index);
            this->labels.push_back(weight);
        }
    }
    this->vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0);
    this->context_uniform_sampler.append(context_uniform_distribution, 1.0);
    this->node_sampler.append(node_distribution, 1.0);
    this->negative_sampler.append(neg_distribution, 0.75);
    this->edge_sampler.append(edge_distribution, 1.0);
    std::cout << "\tDone" << std::endl;
}

std::vector<long> EdgeSampler::draw_an_edge() {
    std::vector<long> edge;
    long index = this->edge_sampler.draw();
    edge.push_back(this->vertexes[index]);
    edge.push_back(this->contexts[index]);
    return edge;
}

void EdgeSampler::feed_an_edge(long* from_node, long* to_node) {
    long index = this->edge_sampler.draw();
    *from_node = this->vertexes[index];
    *to_node = this->contexts[index];
}

long EdgeSampler::draw_a_vertex() {
    return this->draw_an_edge()[0];
}

long EdgeSampler::draw_a_context() {
    return this->draw_an_edge()[1];
}

long EdgeSampler::draw_a_node() {
    return this->node_sampler.draw();
}

long EdgeSampler::draw_a_node_uniformly() {
    return random_range(0, this->edge_size);
}

long EdgeSampler::draw_a_negative() {
    return this->unique_contexts[random_range(0, this->context_size)];
}

long EdgeSampler::draw_a_vertex_uniformly() {
    return this->vertex_uniform_sampler.draw();
}

long EdgeSampler::draw_a_context_uniformly() {
    return this->context_uniform_sampler.draw();
}

std::vector<long> EdgeSampler::get_neighbors(long vertex_index) {
    std::vector<long> neighbors;
    long offset=this->offset[vertex_index];
    long branch=this->branch[vertex_index];
    for (int i=0; i<branch; i++)
        neighbors.push_back(this->contexts[offset+i]);
    return neighbors;
}
