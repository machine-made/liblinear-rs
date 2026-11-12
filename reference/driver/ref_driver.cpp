// SPDX-License-Identifier: Apache-2.0
#include "linear.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void quiet(const char*) {}

void print_double(FILE* out, double value) {
    require(std::isfinite(value), "The result is not finite.");
    std::fprintf(out, "%.17g", value);
}

struct Case {
    std::vector<int> solvers;
    parameter param{};
    double bias = -1;
    int features = 0;
    std::vector<int> weight_labels;
    std::vector<double> class_weights;
    std::vector<double> labels;
    std::vector<std::vector<feature_node>> rows;
};

Case read_case(const char* path) {
    std::ifstream input(path);
    require(bool(input), "Cannot open the case file.");
    Case c;
    c.param.C = 1;
    c.param.eps = 0.01;
    c.param.p = 0.1;
    c.param.nu = 0.5;
    c.param.regularize_bias = 1;
    int row_count = 0;
    bool data = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        if (!data) {
            std::string key;
            fields >> key;
            if (key == "solvers") {
                int solver;
                while (fields >> solver) c.solvers.push_back(solver);
            } else if (key == "C") fields >> c.param.C;
            else if (key == "eps") fields >> c.param.eps;
            else if (key == "bias") fields >> c.bias;
            else if (key == "p") fields >> c.param.p;
            else if (key == "nu") fields >> c.param.nu;
            else if (key == "regularize_bias") fields >> c.param.regularize_bias;
            else if (key == "features") fields >> c.features;
            else if (key == "rows") fields >> row_count;
            else if (key == "weights") {
                int count = 0;
                fields >> count;
                require(count >= 0, "The weight count is not valid.");
                for (int i = 0; i < count; ++i) {
                    int label = 0;
                    double weight = 0;
                    fields >> label >> weight;
                    require(bool(fields) && weight > 0 && std::isfinite(weight),
                            "The class weight is not valid.");
                    c.weight_labels.push_back(label);
                    c.class_weights.push_back(weight);
                }
            } else if (key == "data") data = true;
            else throw std::runtime_error("The case key is not valid.");
            require(!fields.fail() || key == "solvers", "The case value is not valid.");
        } else {
            double label = 0;
            fields >> label;
            require(bool(fields) && std::isfinite(label), "The row label is not valid.");
            c.labels.push_back(label);
            std::vector<feature_node> nodes;
            std::string token;
            int previous = 0;
            while (fields >> token) {
                std::istringstream feature(token);
                int index = 0;
                char colon = 0;
                double value = 0;
                feature >> index >> colon >> value;
                require(bool(feature) && feature.peek() == EOF && colon == ':' &&
                            index > previous && index <= c.features && std::isfinite(value),
                        "The feature is not valid.");
                nodes.push_back({index, value});
                previous = index;
            }
            if (c.bias >= 0) nodes.push_back({c.features + 1, c.bias});
            nodes.push_back({-1, 0});
            c.rows.push_back(std::move(nodes));
        }
    }
    require(data && !c.solvers.empty() && c.features > 0 && row_count > 0 &&
                int(c.rows.size()) == row_count && std::isfinite(c.bias),
            "The case is not complete.");
    return c;
}

void write_run(FILE* out, Case& c, int solver) {
    std::vector<feature_node*> x;
    for (auto& row : c.rows) x.push_back(row.data());
    problem prob{};
    prob.l = int(x.size());
    prob.n = c.features + (c.bias >= 0 ? 1 : 0);
    prob.y = c.labels.data();
    prob.x = x.data();
    prob.bias = c.bias;
    c.param.solver_type = solver;
    c.param.nr_weight = int(c.class_weights.size());
    c.param.weight_label = c.weight_labels.data();
    c.param.weight = c.class_weights.data();
    const char* error = check_parameter(&prob, &c.param);
    if (error) throw std::runtime_error(error);
    std::srand(1);
    model* fitted = train(&prob, &c.param);
    require(fitted != nullptr, "The model is empty.");
    const auto release = [](model* m) { free_and_destroy_model(&m); };
    std::unique_ptr<model, decltype(release)> owned(fitted, release);
    const int columns = fitted->nr_class == 2 && solver != MCSVM_CS ? 1 : fitted->nr_class;
    std::fprintf(out, "{\"solver\":%d,\"C\":", solver);
    print_double(out, c.param.C);
    std::fprintf(out, ",\"eps\":");
    print_double(out, c.param.eps);
    std::fprintf(out, ",\"bias\":");
    print_double(out, fitted->bias);
    std::fprintf(out, ",\"nr_class\":%d,\"nr_feature\":%d,\"rho\":",
                 fitted->nr_class, fitted->nr_feature);
    if (solver == ONECLASS_SVM) print_double(out, fitted->rho);
    else std::fprintf(out, "null");
    std::fprintf(out, ",\"labels\":");
    if (!fitted->label) std::fprintf(out, "null");
    else {
        std::fprintf(out, "[");
        for (int i = 0; i < fitted->nr_class; ++i)
            std::fprintf(out, "%s%d", i ? "," : "", fitted->label[i]);
        std::fprintf(out, "]");
    }
    std::fprintf(out, ",\"weights\":[");
    for (int i = 0; i < prob.n * columns; ++i) {
        if (i) std::fprintf(out, ",");
        print_double(out, fitted->w[i]);
    }
    std::fprintf(out, "],\"decision_values\":[");
    std::vector<double> predictions;
    std::vector<double> decisions(columns);
    for (int i = 0; i < prob.l; ++i) {
        predictions.push_back(predict_values(fitted, prob.x[i], decisions.data()));
        std::fprintf(out, "%s[", i ? "," : "");
        for (int j = 0; j < columns; ++j) {
            if (j) std::fprintf(out, ",");
            print_double(out, decisions[j]);
        }
        std::fprintf(out, "]");
    }
    std::fprintf(out, "],\"predictions\":[");
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (i) std::fprintf(out, ",");
        print_double(out, predictions[i]);
    }
    std::fprintf(out, "]}");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Use: ref_driver CASE.txt OUTPUT.json");
        std::setlocale(LC_ALL, "C");
        set_print_string_function(quiet);
        Case c = read_case(argv[1]);
        FILE* out = std::fopen(argv[2], "wb");
        require(out != nullptr, "Cannot open the output file.");
        const auto close = [](FILE* f) { std::fclose(f); };
        std::unique_ptr<FILE, decltype(close)> owned(out, close);
        std::fprintf(out, "{\"runs\":[");
        for (size_t i = 0; i < c.solvers.size(); ++i) {
            if (i) std::fprintf(out, ",");
            write_run(out, c, c.solvers[i]);
        }
        std::fprintf(out, "]}\n");
        require(std::fflush(out) == 0 && !std::ferror(out), "Cannot write the output file.");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
