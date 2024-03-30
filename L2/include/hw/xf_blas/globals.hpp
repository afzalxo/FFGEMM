#ifndef XF_BLAS_GLOBALS_HPP
#define XF_BLAS_GLOBALS_HPP

#define FACTORS_U {\
			{1, 0, 0, 1}\
		  }

#define FACTORS_V {\
			{1, 0, 0, 1}\
		  }

namespace xf {
namespace blas {

constexpr unsigned int NUM_FACTORS_U[7] = {2, 2, 1, 1, 2, 2, 2};
constexpr unsigned int NUM_FACTORS_V[7] = {2, 1, 2, 2, 1, 2, 2};

constexpr unsigned int INCRE_FACTORS_U[7] = {0, 2, 4, 5, 6, 8, 10};
constexpr unsigned int INCRE_FACTORS_V[7] = {0, 2, 3, 5, 7, 8, 10};

} // namespace blas
} // namespace xf

#endif // XF_BLAS_GLOBALS_HPP
