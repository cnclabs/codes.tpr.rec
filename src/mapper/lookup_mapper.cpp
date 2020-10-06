#include "lookup_mapper.h"

LookupMapper::LookupMapper(int size, int dimension) {
    this->embedding.resize(size);
    this->size = size;
    this->dimension = dimension;
    for (long index=0; index<size; ++index)
    {
        this->embedding[index].resize(dimension);
        for (int d=0; d<dimension; ++d)
        {
            this->embedding[index][d] = (rand()/(double)RAND_MAX - 0.5) / dimension;
        }
    }
}

void LookupMapper::update(long index, std::vector<double>& loss_vector, double alpha) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*loss_vector[d];
    }
}

void LookupMapper::update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*(loss_vector[d] - lambda*embedding[index][d]);
    }
}

void LookupMapper::save_to_file(std::vector<char*>& index2node, std::string file_name) {
    std::cout << "Save Mapper:" << std::endl;
    std::ofstream embedding_file(file_name);
    if (embedding_file)
    {
        //embedding_file << this->size << " " << this->dimension << std::endl;
        for (long index=0; index!=this->size; index++)
        {
            embedding_file << index2node[index];
            embedding_file << " " << embedding[index][0];
            for (int dim=1; dim!=this->dimension; dim++)
            {
                embedding_file << " " << this->embedding[index][dim];
            }
            embedding_file << std::endl;
        }
        std::cout << "\tSave to <" << file_name << ">" << std::endl;
    }
    else
    {
        std::cout << "\tfail to open file" << std::endl;
    }
}

void LookupMapper::save_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append) {
    std::cout << "Save Mapper:" << std::endl;
    std::ofstream embedding_file;
    if (append)
    {
        std::cout << "Append Mapper:" << std::endl;
        embedding_file.open(file_name, std::ios_base::app);
    }
    else
    {
        std::cout << "Save Mapper:" << std::endl;
        embedding_file.open(file_name);
    }

    if (embedding_file)
    {
        for (auto index: indexes)
        {
            embedding_file << file_graph->index2node[index];
            embedding_file << " " << embedding[index][0];
            for (int dim=1; dim!=this->dimension; dim++)
            {
                embedding_file << " " << this->embedding[index][dim];
            }
            embedding_file << std::endl;
        }
        std::cout << "\tSave to <" << file_name << ">" << std::endl;
    }
    else
    {
        std::cout << "\tfail to open file" << std::endl;
    }
}


void LookupMapper::save_trans_to_file(FileGraph* file_graph, std::string file_name) {
    std::cout << "Save Mapper:" << std::endl;
    std::vector<double> fused_embedding(this->dimension, 0.0);
    long branch, from_index, to_index;
    std::ofstream embedding_file(file_name);
    if (embedding_file)
    {
        for (from_index=0; from_index<file_graph->index2node.size(); from_index++)
        {
            branch = file_graph->index_graph[from_index].size();
            embedding_file << file_graph->index2node[from_index];
            if (branch > 0)
            {
                fused_embedding.assign(this->dimension, 0.0);
                for (auto it: file_graph->index_graph[from_index])
                {
                    to_index = it.first;
                    for (int dim=0; dim!=this->dimension; dim++)
                    {
                        fused_embedding[dim] += this->embedding[from_index][dim];
                        fused_embedding[dim] += this->embedding[to_index][dim];
                    }
                }
                embedding_file << "\t" << fused_embedding[0]/branch;
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << fused_embedding[dim]/branch;
                }
                embedding_file << std::endl;
            }
            else
            {
                embedding_file << "\t" << embedding[from_index][0];
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << this->embedding[from_index][dim];
                }
                embedding_file << std::endl;
            }
        }
        std::cout << "\tSave to <" << file_name << ">" << std::endl;
    }
    else
    {
        std::cout << "\tfail to open file" << std::endl;
    }
}

void LookupMapper::save_gcn_to_file(FileGraph* file_graph, std::vector<long> indexes, std::string file_name, int append) {
    std::vector<double> fused_embedding(this->dimension, 0.0);
    long branch, from_index, to_index;
    double weight, weight_sum;
    std::ofstream embedding_file;
    if (append)
    {
        std::cout << "Append Mapper:" << std::endl;
        embedding_file.open(file_name, std::ios_base::app);
    }
    else
    {
        std::cout << "Save Mapper:" << std::endl;
        embedding_file.open(file_name);
    }

    if (embedding_file)
    {
        for (auto from_index: indexes)
        {
            branch = file_graph->index_graph[from_index].size();
            if (branch > 0)
            {
                embedding_file << file_graph->index2node[from_index];
                fused_embedding.assign(this->dimension, 0.0);
                weight_sum = 0.0;
                for (auto it: file_graph->index_graph[from_index])
                {
                    to_index = it.first;
                    weight = file_graph->index_graph[from_index][to_index];
                    weight_sum += weight;
                    for (int dim=0; dim!=this->dimension; dim++)
                    {
                        fused_embedding[dim] += this->embedding[to_index][dim]*weight;
                    }
                }
                embedding_file << "\t" << (this->embedding[from_index][0]+fused_embedding[0]/weight_sum)/2.0;
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << (this->embedding[from_index][dim]+fused_embedding[dim]/weight_sum)/2.0;
                }
                embedding_file << std::endl;
            }
            else
            {
                embedding_file << file_graph->index2node[from_index];
                embedding_file << "\t" << this->embedding[from_index][0];
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << this->embedding[from_index][dim];
                }
                embedding_file << std::endl;
            }
        }
        std::cout << "\tSave to <" << file_name << ">" << std::endl;
    }
    else
    {
        std::cout << "\tfail to open file" << std::endl;
    }
    embedding_file.close();
}


std::vector<double>& LookupMapper::operator[](long index) {
    return this->embedding[index];
}

std::vector<double> LookupMapper::avg_embedding(std::vector<long>& indexes) {
    std::vector<double> avg_embedding(this->dimension, 0.0);
    for (auto index: indexes)
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[index][d];
    return avg_embedding;
}

std::vector<double> LookupMapper::textgcn_embedding(std::vector<long>& indexes) {
    std::vector<double> avg_embedding(this->dimension, 0.0);
    double size = indexes.size()-1;
    if (size)
    {
        for (auto it=++indexes.begin(); it!=indexes.end(); it++)
            for (int d=0; d<this->dimension; d++)
                avg_embedding[d] += this->embedding[*(it)][d];
        for (int d=0; d<this->dimension; d++)
        {
            avg_embedding[d] = (this->embedding[indexes[0]][d] + avg_embedding[d]/size)/2.0;
        }
    }
    else{
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[indexes[0]][d];
    }
    return avg_embedding;
}


