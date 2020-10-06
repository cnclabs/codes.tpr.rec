#ifndef HASH_H
#define HASH_H
#include "hash.h"

n2iHash::n2iHash() {
    this->HASH_TABLE_SIZE = 30000000;
    this->table.resize(HASH_TABLE_SIZE, -1);
}

n2iHash::n2iHash(long table_size) {
    this->HASH_TABLE_SIZE = table_size;
    this->table.resize(this->HASH_TABLE_SIZE, -1);
}

unsigned int n2iHash::BKDRn2iHash(char *key) {
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;
    while (*key)
    {
        hash = hash * seed + (*key++);
    }
    return (hash % HASH_TABLE_SIZE);
}

void n2iHash::insert_key(char *key) {
    unsigned int pos = this->BKDRn2iHash(key);
    while (this->table[pos] != -1)
        pos = (pos + 1) % this->HASH_TABLE_SIZE;
    this->table[pos] = this->keys.size();
    this->keys.push_back(strdup(key));
}

long n2iHash::search_key(char *key) {
    unsigned int pos = this->BKDRn2iHash(key);
    while (1)
    {
        if (this->table[pos] == -1)
            return -1;
        if ( !strcmp(key, this->keys[ this->table[pos] ]) )
            return this->table[pos];
        pos = (pos + 1) % this->HASH_TABLE_SIZE;
    }
}

#endif
