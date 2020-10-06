#include <vector>
#include <string.h>

class n2iHash {
    /* Custom hash table
     */
    private:
        long HASH_TABLE_SIZE;
        unsigned int BKDRn2iHash(char* key);

    public:
        n2iHash();
        n2iHash(long table_size);

        // variables
        std::vector< long > table; // init by -1
        std::vector< char* > keys;

        // operations
        void insert_key(char *key);
        long search_key(char *key);
};
