#ifndef LOOKUP_MAPPER_H
#define LOOKUP_MAPPER_H
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../util/file_graph.h"

class LookupMapper {
    public:
        //variable
        int size, dimension;
        std::vector<std::vector<double>> embedding;

        // embedding function
        std::vector<double> avg_embedding(std::vector<long>& indexes);
        std::vector<double> textgcn_embedding(std::vector<long>& indexes);

        // constructor
        LookupMapper(int size, int dimension);

        // update function
        void update(long index, std::vector<double>& loss_vector, double alpha);
        void update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda);

        // save function
        void save_to_file(std::vector<char*>& index2vertex, std::string file_name);
        void save_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);
        void save_trans_to_file(FileGraph* file_graph, std::string file_name);
        void save_gcn_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append);

        // overload operator
        std::vector<double>& operator[](long index);
};
#endif
