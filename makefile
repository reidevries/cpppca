##
# cpppca
#
# @file
# @version 0.1

CC=g++
EIGENDIR=/usr/include/eigen3

all:
	$(CC) -I$(EIGENDIR) main.cpp -o cpppca

# end
