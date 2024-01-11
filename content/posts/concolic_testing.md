+++
title = 'Concolic Testing With KLEE'
date = 2014-07-08T16:17:54+01:00
draft = false
+++

## Introduction to Automated Software Testing

Software testing is a complex and challenging task. While complete automation of this process remains a distant goal, there are promising techniques emerging in the field. One such technique is Concolic Testing, which offers a semi-automated approach to software testing.

### Understanding Concolic Testing

Concolic Testing is an innovative method that blends concrete and symbolic execution to generate test cases. The term "Concolic" itself is a blend of 'Concrete' and 'Symbolic.' This technique aims to explore as many execution paths as possible within a program, creating symbolic constraints along these paths. These constraints are then solved, and if feasible, the resulting values are used as input for the program. These inputs effectively act as test cases, assuming the program behaves deterministically.

### Implementing Concolic Testing with KLEE

To demonstrate Concolic Testing, we will use KLEE, a prominent Concolic Testing engine. We'll test a basic implementation of Euclid's algorithm for finding the greatest common divisor (gcd). 

```c
#include<assert.h>

int gcd(int x, int y) {
    while (x != y) {
        if (x > y)
            x = x - y;
        else
            y = y - x;
    }
    return x;
}

int main(int argc, char **argv) {
    int x, y, g;
    scanf("%d", &x);
    scanf("%d", &y);

    if(x > 0 && y != 0) {
        g = gcd(x, y);
        assert(x % g == 0);
        assert(y % g == 0);
    }
}
```

In this case, we tag the variables `x` and `y` as symbolic since they determine the inputs for the `gcd` function. We replace the `scanf` commands with KLEE-specific ones to achieve this.

```c
[...]
int x, y, g;

klee_make_symbolic(&x, sizeof(int), "x");
klee_make_symbolic(&y, sizeof(int), "y");

if(x > 0 && y != 0) {
[...]
```

The next step is compiling the code into LLVM bitcode and passing it to KLEE.

```bash
llvm-gcc --emit-llvm -c -g gcd.c -o gcd.o 
klee -max-time=120 -only-output-states-covering-new gcd.o
```

### Evaluating Test-Case Suites

To assess the effectiveness of our test suite, we use the `gcov` tool from the GNU Compiler Collection. We compile our program with GCC, including flags for gcov and KLEE.

```cmake
override CFLAGS += -g -fprofile-arcs -ftest-coverage
app: gcd.c
    gcc -L /your/path/to/klee/Release+Asserts/lib/ $(CFLAGS) -o gcd gcd.c -lkleeRuntest
```

After running the makefile and executing the tests, we apply `gcov` to review the code coverage.

```bash
make
export LD_LIBRARY_PATH=/your/path/to/klee/Release+Asserts/lib/:$LD_LIBRARY_PATH
for i in klee-last/*.ktest; do KTEST_FILE=$i ./gcd; done

gcov gcd
```

Our tests achieved 100% code coverage, a promising indicator of thorough testing. However, it's important to note that full coverage doesn't guarantee the absence of bugs.

### Concluding Thoughts

This post aimed to provide an overview of Concolic Testing and its application through KLEE. While still a subject of research and not fully matured, tools like KLEE demonstrate significant potential in automated software testing, achieving high code coverage in complex software like the GNU Coreutils.
