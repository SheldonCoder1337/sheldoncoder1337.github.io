# Cook with C Language

C programming is like cooking in a kitchen: you have recipes (header file declarations), specific cooking steps, the amount of seasoning (source code definitions), and meal preparation steps (Makefile). Let's illustrate with a calculator example:

## Recipes: Header File

The recipe (header file) declares the kitchenware or seasoning: what functions and data structures we will use.

```c title="calc.h"
```c title="calc.h"
/*
 * Calculator implementation plan:
 * 
 * We want to implement a function compute(op, a, b) that performs
 * basic arithmetic (add, subtract, multiply, divide).
 * 
 * To achieve this, we need to:
 *   - Parse user input to extract the operation and operands.
 *   - Validate whether the parsed expression is well-formed.
 *   - Then compute the result using the parsed values.
 * 
 * Required components:
 * 1. An enumeration of operation types: op_t {OP_ADD, OP_SUB, OP_MUL, OP_DIV}
 * 2. A structure to hold a parsed expression: expr_t {op, a, b, valid (int)}
 * 3. A function to parse a string into an expr_t: parse_expression(const char *input)
 * 4. A function to compute the result from operation and operands: compute(op_t op, double a, double b, int *error)
 */

#ifndef CALC_H
#define CALC_H

typedef enum {
    CALC_OP_ADD,    // Addition
    CALC_OP_SUB,    // Subtraction
    CALC_OP_MUL,    // Multiplication
    CALC_OP_DIV     // Division
} calc_op_t; 

// POSIX convention: type definitions use _t suffix, prefixed with module name to avoid global namespace collisions

typedef struct {
    calc_op_t op;   // Operator
    double a;       // Left operand
    double b;       // Right operand
    int valid;      // 1 if parse succeeded, 0 if failed
} calc_expr_t;

/*
 * Parse a string expression and return a calc_expr_t.
 * Parameters: input - null-terminated string, must not be NULL.
 * Returns: parsed expression; caller must check the valid field.
 */
calc_expr_t calc_parse_expression(const char *input);

/*
 * Perform the computation.
 * Parameters: op - operator, a, b - operands, error - output error flag (may be NULL).
 * Returns: result of the operation; if error is non-NULL and an error occurs (e.g., division by zero), *error is set to 1.
 */
double calc_compute(calc_op_t op, double a, double b, int *error);

#endif
```

## Cook in Detail: Source Code

Now we prepare the ingredients and cook step by step. Each `.c` file implements a specific part of the recipe.

```c title="parser.c"
#include "calc.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* Static function: visible only in this file, used to check input validity */
static int is_valid_input(const char *input) {
    return input != NULL && input[0] != '\0';
}

/* Static function: skip leading whitespace (internal use) */
static const char *skip_whitespace(const char *s) {
    while (s && isspace((unsigned char)*s)) s++;
    return s;
}

calc_expr_t calc_parse_expression(const char *input) {
    calc_expr_t result = {0, 0.0, 0.0, 0};  // Initialize to invalid

    // Defensive programming: check input pointer
    if (!is_valid_input(input)) {
        return result;  // Invalid input, return immediately
    }

    // Skip leading whitespace to avoid parsing issues
    const char *p = skip_whitespace(input);

    char op_char;
    int n = sscanf(p, "%lf %c %lf", &result.a, &op_char, &result.b);
    if (n != 3) {
        return result;  // Parsing failed, valid remains 0
    }

    // Set enumeration value based on operator character
    switch (op_char) {
        case '+': result.op = CALC_OP_ADD; break;
        case '-': result.op = CALC_OP_SUB; break;
        case '*': result.op = CALC_OP_MUL; break;
        case '/': result.op = CALC_OP_DIV; break;
        default:
            return result;  // Unsupported operator
    }

    result.valid = 1;  // Parsing succeeded
    return result;
}
```

*The parser reads the user input string, extracts the operator and operands, and returns an expression structure.*

```c title="compute.c"
#include "calc.h"

double calc_compute(calc_op_t op, double a, double b, int *error) {
    // Defensive programming: if error is not NULL, set it to 0 initially
    if (error) {
        *error = 0;
    }

    switch (op) {
        case CALC_OP_ADD: return a + b;
        case CALC_OP_SUB: return a - b;
        case CALC_OP_MUL: return a * b;
        case CALC_OP_DIV:
            if (b == 0.0) {
                if (error) {
                    *error = 1;  // Division by zero error
                }
                return 0.0;
            }
            return a / b;
        default:
            // Unknown operator, also treat as error
            if (error) {
                *error = 1;
            }
            return 0.0;
    }
}
```

*This source file implements the actual computation logic, handling each operation and error conditions (like division by zero).*

```c title="main.c"
#include "calc.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // Check number of command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s \"a + b\"\n", argv[0]);
        return 1;
    }

    // Parse expression
    calc_expr_t expr = calc_parse_expression(argv[1]);
    if (!expr.valid) {
        fprintf(stderr, "Invalid expression: %s\n", argv[1]);
        return 1;
    }

    // Compute result
    int error;
    double result = calc_compute(expr.op, expr.a, expr.b, &error);
    if (error) {
        fprintf(stderr, "Computation error (division by zero?)\n");
        return 1;
    }

    // Output result
    printf("%.2f\n", result);
    return 0;
}
```

*The main function orchestrates the process: it takes input from the command line, parses it, computes the result, and prints it or reports errors.*

## Meal Prep: Makefile

The Makefile automates the build process, defining how to compile and link the source files into an executable. It also includes a clean target to remove generated files.

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -g
TARGET = calc
OBJDIR = obj
SOURCES = src/main.c src/parser.c src/compute.c
OBJECTS = $(SOURCES:src/%.c=$(OBJDIR)/%.o)

$(TARGET): $(OBJECTS)
    $(CC) $^ -o $@

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
    $(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
    mkdir -p $(OBJDIR)

clean:
    rm -rf $(OBJDIR) $(TARGET)

.PHONY: clean
```

### Building and Running

When we run `make`, the compiler commands are executed as shown below:

```bash
gcc -Wall -Wextra -Iinclude -g    -c src/main.c -o obj/main.o
gcc -Wall -Wextra -Iinclude -g    -c src/parser.c -o obj/parser.o
gcc -Wall -Wextra -Iinclude -g    -c src/compute.c -o obj/compute.o
gcc obj/main.o obj/parser.o obj/compute.o -o calc
```

Now we can test our calculator:

```bash
$ ./calc "3 + 4"
7.00
$ ./calc "10 / 0"
Computation error (division by zero?)
$ ./calc "5 * 2"
10.00
```

### Debugging with GDB

We can also use a debugger (like GDB) to inspect the program's behavior. For example, setting a breakpoint in `parse_expression`:

```bash
$ gdb ./calc
(gdb) break calc_parse_expression
Breakpoint 1 at 0x13b1: file src/parser.c, line 17.
(gdb) run "2 * 3"
Breakpoint 1, calc_parse_expression (input=0x7fffffffdfa2 "2 * 3") at src/parser.c:17
17      calc_expr_t calc_parse_expression(const char *input) {
(gdb) next
18          calc_expr_t result = {0, 0.0, 0.0, 0};  // Initialize to invalid
(gdb) next
21          if (!is_valid_input(input)) {
(gdb) print op_char
$1 = 0 '\000'
(gdb) next
26          const char *p = skip_whitespace(input);
(gdb) print n
$2 = 0
(gdb) continue
Continuing.
6.00
[Inferior 1 (process 1710) exited normally]
```

This shows how we can step through the code and examine variables at runtime.

---

Just like a well-organized kitchen, a well-structured C program separates declarations (recipes), implementations (cooking steps), and build automation (meal prep).