CC = g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast
UTIL_OBJECTS = util random hash file_graph
SAMPLER_OBJECTS = alias_methods vc_sampler edge_sampler
MAPPER_OBJECTS = lookup_mapper
OPTIMIZER_OBJECTS = pair_optimizer triplet_optimizer quadruple_optimizer
HUB_CLIS = tpr
LIBS= -L ./ -lsmore

all: $(UTIL_OBJECTS) $(SAMPLER_OBJECTS) $(MAPPER_OBJECTS) $(OPTIMIZER_OBJECTS) $(HUB_CLIS)

%.o: %.cpp %.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

$(UTIL_OBJECTS):
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o src/util/$@.o src/util/$@.cpp
	ar rcs ./libsmore.a src/util/$@.o

$(SAMPLER_OBJECTS):
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o src/sampler/$@.o src/sampler/$@.cpp
	ar rcs ./libsmore.a src/sampler/$@.o

$(MAPPER_OBJECTS):
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o src/mapper/$@.o src/mapper/$@.cpp
	ar rcs ./libsmore.a src/mapper/$@.o

$(OPTIMIZER_OBJECTS):
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o src/optimizer/$@.o src/optimizer/$@.cpp
	ar rcs ./libsmore.a src/optimizer/$@.o

$(HUB_CLIS):
	$(CC) $(CPPFLAGS) hub/$@.cpp $(LIBS) -o $@

clean:
	rm -f src/util/*.o
	rm -f src/sampler/*.o
	rm -f src/mapper/*.o
	rm -f src/optimizer/*.o
	rm -f $(HUB_CLIS)
	rm -f ./libsmore.a
