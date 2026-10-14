// Alexander duality turns finite H1 of the descending primal filtration into
// component merges of ascending dual faces. The exterior is always oldest.
// No Python or third-party headers, and no persistent mutable native state.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace {
std::int64_t root(std::vector<std::int64_t>& parent, std::int64_t node) {
    while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    return node;
}
}

extern "C" std::int64_t superpov_planar_pairs(
    std::int64_t face_count, std::int64_t edge_count,
    const std::int64_t* triangles, const std::int64_t* edges,
    const std::int64_t* neighbors, const double* values,
    double min_persistence, double* output) noexcept {
    try {
        std::vector<std::int64_t> parent(face_count + 1);
        std::iota(parent.begin(), parent.end(), 0);
        std::vector<double> birth(face_count + 1);
        for (std::int64_t face = 0; face < face_count; ++face) {
            birth[face] = std::min(values[triangles[3 * face]],
                                  std::min(values[triangles[3 * face + 1]],
                                           values[triangles[3 * face + 2]]));
        }
        birth[face_count] = -std::numeric_limits<double>::infinity();
        std::vector<double> weights(edge_count);
        std::vector<std::int64_t> order(edge_count);
        std::iota(order.begin(), order.end(), 0);
        for (std::int64_t edge = 0; edge < edge_count; ++edge) {
            weights[edge] = std::min(values[edges[2 * edge]], values[edges[2 * edge + 1]]);
        }
        std::sort(order.begin(), order.end(), [&weights](std::int64_t left, std::int64_t right) {
            return weights[left] < weights[right]
                || (weights[left] == weights[right] && left < right);
        });
        std::vector<std::pair<double, double> > pairs;
        for (const auto edge : order) {
            auto older = root(parent, neighbors[2 * edge]);
            auto younger = root(parent, neighbors[2 * edge + 1]);
            if (older == younger) continue;
            if (birth[older] > birth[younger]
                || (birth[older] == birth[younger] && older > younger)) {
                std::swap(older, younger);
            }
            const double high = weights[edge];
            const double low = birth[younger];
            if (high - low > min_persistence) pairs.emplace_back(high, low);
            parent[younger] = older;
        }
        std::stable_sort(pairs.begin(), pairs.end(),
            [](const std::pair<double, double>& left, const std::pair<double, double>& right) {
                return left.first - left.second > right.first - right.second;
            });
        for (std::size_t index = 0; index < pairs.size(); ++index) {
            output[2 * index] = pairs[index].first;
            output[2 * index + 1] = pairs[index].second;
        }
        return static_cast<std::int64_t>(pairs.size());
    } catch (...) {
        return -1;
    }
}
