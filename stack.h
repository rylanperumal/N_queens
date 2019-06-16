#ifndef STACK
#define STACK

#include <stdlib.h>
#include <string.h>

struct Stack{
  int top;
  unsigned int capacity;
  int *array;
};
#endif
