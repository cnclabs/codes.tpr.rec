#include "util.h"

ArgParser::ArgParser(int argc, char** argv) {
    this->argc = argc;
    this->argv = argv;
    std::cout << "Parse Arguments:" << std::endl;
}

int ArgParser::get_int(std::string flag, int value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << atoi(argv[a+1]);
            std::cout << " (" << description << ")" << std::endl;
            return atoi(argv[a+1]);
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}

double ArgParser::get_double(std::string flag, double value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << atof(argv[a+1]);
            std::cout << " (" << description << ")" << std::endl;
            return atof(argv[a+1]);
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}

std::string ArgParser::get_str(std::string flag, std::string value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << argv[a+1];
            std::cout << " (" << description << ")" << std::endl;
            return argv[a+1];
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}


int is_directory(std::string path) {
    struct stat info;
    if( stat( path.c_str(), &info ) != 0 ) // nothing
        return -1;
    else if( info.st_mode & S_IFDIR ) // a folder
        return 1;
    return 0; // a file
}

double dot_similarity(std::vector<double>& embeddingA, std::vector<double>& embeddingB, int dimension) {
    double prediction=0;
    for (int d=0; d<dimension; d++)
    {
        prediction += embeddingA[d]*embeddingB[d];
    }
    return prediction;
}

Monitor::Monitor(unsigned long long total_step) {
    this->total_step = total_step;
}

void Monitor::progress(unsigned long long* current_step) {
    printf("\tProgress:\t%.3f %%%c", (double)*current_step/this->total_step*100.0, 13);
    fflush(stdout);
}

void Monitor::end() {
    printf("\tProgress:\t%.3f\n", 100.0);
}

