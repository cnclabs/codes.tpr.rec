#ifndef BASE_SAMPLER_H
#define BASE_SAMPLER_H
#include <string>
#include <string.h>
#include <unordered_map>
#include <set>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include "hash.h"
#include "util.h"

#define SAMPLER_MONITOR 10000

typedef std::unordered_map<long, std::unordered_map<long, double>> IndexGraph;

class FileGraph {
    /* FileGraph loads file-based data as a graph.
     */
    private:
        // helper variables / functions
        std::vector<std::string> file_names;
        std::vector<unsigned long long> file_lines;
        void load_file_status(std::string path);
        void inherit_index2node(std::vector<char*>&);

        // load from files
        void load_from_edge_list(std::string path, int undirected);

        // TODO: implement other ways to read from grpah files
        //void load_from_adjacency_list(std::string path);

    public:
        // constuctor
        FileGraph(std::string path, int undirected);
        FileGraph(std::string path, int undirected, std::vector<char*>& index2node);

        // func
        std::vector<long> get_all_nodes();
        std::vector<long> get_all_from_nodes();
        std::vector<long> get_all_to_nodes();

        // graph-related variables
        long edge_size=0;
        n2iHash node2index;
        std::vector<char*> index2node;
        IndexGraph index_graph;
};
#endif
