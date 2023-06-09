TRAIN_SET_SIZE := 500 1000 2000
NEURONS_NUM := 50 100 200 500 1000 # neurons in hidden layer
THREADS_NUM := 1 2 4 12

DEFAULT_TRAIN_SIZE := 1000
DEFAULT_THR := 4
DEFAULT_NEURONS := 500

main:
	g++ -Wall -o main.out src/main.cpp -fopenmp

run:
	./main.out $(DEFAULT_TRAIN_SIZE) $(DEFAULT_THR) $(DEFAULT_NEURONS)

test: # testing duration times
	mkdir -p times/
	mkdir -p plots/

	for train_size in $(TRAIN_SET_SIZE); do \
		mkdir -p times/$$train_size; \
		for thr in $(THREADS_NUM); do \
			./main.out $$train_size $$thr $(NEURONS_NUM); \
		done; \
	done
	
	python3 plotting.py