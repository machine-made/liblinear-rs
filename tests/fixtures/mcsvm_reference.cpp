// SPDX-License-Identifier: Apache-2.0
// Use the frozen reader and writer for the six extra multiclass problems.
#define main frozen_driver_main
#include "../../reference/driver/ref_driver.cpp"
#undef main
#include <cstring>

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Use: mcsvm_reference CASE OUTPUT");
        std::setlocale(LC_ALL, "C");
        set_print_string_function(quiet);
        Case c = read_case(argv[1]);
        FILE* out = std::fopen(argv[2], "wb");
        require(out != nullptr, "Cannot open the output file.");
        std::fprintf(out, "{\"runs\":[");
        for (size_t i = 0; i < c.solvers.size(); ++i) {
            require(c.solvers[i] == 4, "Select solver 4.");
            if (i) std::fprintf(out, ",");
            write_run(out, c, c.solvers[i]);
        }
        std::fprintf(out, "]}\n");
        require(std::fclose(out) == 0, "Cannot close the output file.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
