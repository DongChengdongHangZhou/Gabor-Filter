#CXX = g++
#CFLAG =
#LFLAG =

_libfp.so: ridge_filter.o
	g++ -O3 -shared -fPIC -o _libfp.so ridge_filter.o

ridge_filter.o: ridge_filter.cpp
	g++ -O3 -fPIC -c ridge_filter.cpp -o ridge_filter.o
