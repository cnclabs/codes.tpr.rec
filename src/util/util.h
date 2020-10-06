#ifndef UTIL_H
#define UTIL_H
#include <sys/stat.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>

int is_directory(std::string path);
double dot_similarity(std::vector<double>& embeddingA, std::vector<double>& embeddingB, int dimension);

class ArgParser {
    private:
        int argc;
        char** argv;

    public:
        ArgParser(int argc, char** argv);

        int get_int(std::string flag, int value, std::string description);
        double get_double(std::string flag, double value, std::string description);
        std::string get_str(std::string flag, std::string value, std::string description);
};

class Monitor {
    public:
        unsigned long long total_step;

        // constructor
        Monitor(unsigned long long total_step);

        // count
        void progress(unsigned long long* current_step);
        void end();
};

#endif
