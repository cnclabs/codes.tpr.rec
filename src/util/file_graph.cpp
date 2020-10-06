#include "file_graph.h"

FileGraph::FileGraph(std::string path, int undirected) {
    this->load_from_edge_list(path, undirected);
}

FileGraph::FileGraph(std::string path, int undirected, std::vector<char*>& index2node) {
    this->inherit_index2node(index2node);
    this->load_from_edge_list(path, undirected);
}

void FileGraph::load_file_status(std::string path) {
    /* Get file names and lines.
     */

    // get file names
    int _is_directory = is_directory(path);
    if (_is_directory==-1) // fail
    {
        std::cout << "cannot access " << path << std::endl;
        exit(0);
    }
    else if (_is_directory==1) // folder with multiple files
    {
        DIR *dir;
        struct dirent *ent;
        dir = opendir(path.c_str());
        while ((ent = readdir (dir)) != NULL) {
            std::string fname = path + "/" + ent->d_name;
            this->file_names.push_back(fname);
        }
        closedir(dir);
    }
    else // single file
    {
        this->file_names.push_back(path.c_str());
    }

    // get lines
    FILE *fin;
    char c_line[1000];
    unsigned long long num_lines = 0;
    std::cout << "Lines Preview:" << std::endl;
    for (auto fname: this->file_names)
    {
        fin = fopen(fname.c_str(), "rb");
        while (fgets(c_line, sizeof(c_line), fin))
        {
            if (num_lines % SAMPLER_MONITOR == 0)
            {
                printf("\t# of lines:\t%lld%c", num_lines, 13);
            }
            ++num_lines;
        }
        fclose(fin);
        this->file_lines.push_back(num_lines);
        this->edge_size += num_lines;
        printf("\t# of lines:\t%lld%c\n", num_lines, 13);
    }
}

void FileGraph::load_from_edge_list(std::string path, int undirected) {
    this->load_file_status(path);
    std::cout << "Loading Lines:" << std::endl;
    FILE *fin;
    char from_node[256], to_node[256];
    long from_index, to_index;
    double weight;
    unsigned long long line = 0;

    // read from file list
    for (int i=0; i<this->file_names.size(); i++)
    {
        fin = fopen(this->file_names[i].c_str(), "rb");
        for (; line != this->file_lines[i]; line++)
        {
            // read from (node, node, weight)
            if ( fscanf(fin, "%[^\t]\t%[^\t]\t%lf\n", from_node, to_node, &weight) != 3 )
            {
                std::cout << "\t[WARNING] skip line " << line << std::endl;
                continue;
            }
            if (line % SAMPLER_MONITOR == 0)
            {
                printf("\tProgress:\t%.2f %%%c", (double)(line)/(this->file_lines[this->file_names.size()-1]) * 100, 13);
                fflush(stdout);
            }

            // generate index map
            from_index = this->node2index.search_key(from_node);
            if (from_index == -1)
            {
                this->node2index.insert_key(from_node);
                from_index = this->node2index.search_key(from_node);
                this->index2node.push_back(strdup(from_node));
            }
            to_index = this->node2index.search_key(to_node);
            if (to_index == -1)
            {
                this->node2index.insert_key(to_node);
                to_index = this->node2index.search_key(to_node);
                this->index2node.push_back(strdup(to_node));
            }

            // store in index graph
            this->index_graph[from_index][to_index] = weight;
            if (undirected)
            {
                this->index_graph[to_index][from_index] = weight;
            }
        }
    }
    std::cout << "\tProgress:\t100.00 %\r" << std::endl;
    std::cout << "\t# of node:\t" << this->index2node.size() << std::endl;
}

std::vector<long> FileGraph::get_all_nodes() {
    std::unordered_map<long, int> keys;
    std::vector<long> nodes;
    for (auto kv : this->index_graph) {
        keys[kv.first] = 1;
        for (auto v: kv.second)
            keys[v.first] = 1;
    }
    for (auto kv: keys)
        nodes.push_back(kv.first);
    return nodes;
}

std::vector<long> FileGraph::get_all_from_nodes() {
    std::vector<long> nodes;
    for (auto kv : this->index_graph)
        if (kv.second.size())
            nodes.push_back(kv.first);
    return nodes;
}

std::vector<long> FileGraph::get_all_to_nodes() {
    std::unordered_map<long, int> keys;
    std::vector<long> nodes;
    for (auto kv : this->index_graph)
        for (auto v: kv.second)
            keys[v.first] = 1;
    for (auto kv: keys)
        nodes.push_back(kv.first);
    return nodes;
}

void FileGraph::inherit_index2node(std::vector<char*>& index2node) {
    std::cout << "\tinheriting the node map" << std::endl;
    for (int index=0; index<index2node.size(); index++)
    {
        this->node2index.insert_key(index2node[index]);
        this->index2node.push_back(strdup(index2node[index]));
    }
}

