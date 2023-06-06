TRAIN_SET_SIZE := 100 200 300 500 700 800 1000
NEURONS_NUM := 10 30 50 100 150 200 # neurons in hidden layer

main:
	g++ -Wall -o main.out src/main.cpp -fopenmp

run:
	./main.out

test: # testing duration times
	mkdir -p times/
	mkdir -p plots/

	for train_size in $(TRAIN_SET_SIZE); do \
		./main.out $$train_size $(NEURONS_NUM); \
	done

	python3 plotting.py