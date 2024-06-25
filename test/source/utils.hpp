#include <numeric>

constexpr float mseThreshold = 1.0;

template<typename C1, typename C2, typename T = typename C1::value_type>
auto meanSquaredError(const C1& input1, const C2& input2) -> T {
    return std::transform_reduce(input1.cbegin(),
                                 input1.cend(),
                                 input2.cbegin(),
                                 static_cast<T>(0),
                                 std::plus<>(),
                                 [](auto val1, auto val2) {
                                     const auto error = val1 - val2;
                                     return error * error;
                                 })
        / static_cast<T>(input1.size());
}
