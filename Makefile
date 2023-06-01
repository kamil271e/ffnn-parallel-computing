TRAIN_SET_SIZE := 100 200 300 500 700 800 1000

main:
	g++ -Wall -o main.out src/main.cpp

run:
	./main.out

test: # testing duration times
	mkdir -p times/
	mkdir -p plots/

	for train_size in $(TRAIN_SET_SIZE); do \
		./main.out $$train_size 10 30 50 100 150 200; \
	done

	python3 plotting.py

# numbers of neurons in hidden layer are passed as an argv after train set size