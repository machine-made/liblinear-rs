// SPDX-License-Identifier: BSD-3-Clause
// Use the frozen reader and writer with the weight rebuild flag set by argv.
#define main frozen_driver_main
#include "../../reference/driver/ref_driver.cpp"
#undef main
#include <cstring>

namespace {
int fallbacks = 0;
void count_fallbacks(const char* message) {
    if (std::strstr(message, "Switching to use -s 2")) ++fallbacks;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "Use: dual_reference CASE OUTPUT RECALC");
        std::setlocale(LC_ALL, "C");
        set_print_string_function(count_fallbacks);
        Case c = read_case(argv[1]);
        c.param.w_recalc = std::strcmp(argv[3], "1") == 0;
        FILE* out = std::fopen(argv[2], "wb");
        require(out != nullptr, "Cannot open the output file.");
        std::fprintf(out, "{\"runs\":[");
        for (size_t i = 0; i < c.solvers.size(); ++i) {
            if (i) std::fprintf(out, ",");
            write_run(out, c, c.solvers[i]);
        }
        std::fprintf(out, "]}\n");
        require(std::fclose(out) == 0, "Cannot close the output file.");
        std::fprintf(stderr, "Newton fallbacks: %d\n", fallbacks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
