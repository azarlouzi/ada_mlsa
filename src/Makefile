C = g++ -std=c++17
ifeq ($(version), debug)
	C += -g -Wall -Wextra -Wconversion -Wunreachable-code -I.
else
	C += -O2 -I.
endif

all: main clean
main: main.o ml_sa.o nested_sa.o sa.o option_model.o swap_model.o helpers.o ada_nested_sa.o ada_ml_sa.o adaptivity.o
	$C -o main main.o ml_sa.o nested_sa.o sa.o option_model.o swap_model.o helpers.o ada_nested_sa.o ada_ml_sa.o adaptivity.o
main.o: main.cpp helpers.h ml_sa.h nested_sa.h sa.h option_model.h swap_model.h ada_nested_sa.h ada_ml_sa.h
	$C -c main.cpp
swap_model.o: swap_model.cpp swap_model.h ml_sa.h nested_sa.h sa.h helpers.h adaptivity.h
	$C -c swap_model.cpp
option_model.o: option_model.cpp option_model.h ml_sa.h nested_sa.h sa.h helpers.h adaptivity.h
	$C -c option_model.cpp
ada_nested_sa.o: ml_sa.cpp ml_sa.h nested_sa.h sa.h helpers.h adaptivity.h
	$C -c ada_nested_sa.cpp
ada_ml_sa.o: ml_sa.cpp ml_sa.h nested_sa.h sa.h helpers.h adaptivity.h
	$C -c ada_ml_sa.cpp
ml_sa.o: ml_sa.cpp ml_sa.h nested_sa.h sa.h helpers.h
	$C -c ml_sa.cpp
nested_sa.o: nested_sa.cpp nested_sa.h helpers.h sa.h
	$C -c nested_sa.cpp
sa.o: sa.cpp sa.h helpers.h
	$C -c sa.cpp
adaptivity.o: adaptivity.cpp helpers.h
	$C -c adaptivity.cpp
helpers.o: helpers.cpp helpers.h
	$C -c helpers.cpp
clean:
	mv main ../bin
	rm -f *o
