TRAIN_SET_SIZE := 100 250 500 750 1000

main:
	g++ -Wall -o main.out src/main.cpp

run:
	./main.out

test: # testing duration times
	mkdir -p times/

	for train_size in $(TRAIN_SET_SIZE); do \
		./main.out $$train_size 10 30 50 100 200; \
	done
# numbers of neurons in hidden layer are passed as an argv after train set size