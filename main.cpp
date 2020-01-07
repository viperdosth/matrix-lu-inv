#include "Mat.hpp"

#ifdef GPROF
#include <gperftools/profiler.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

int main(int argc, char const *argv[]) {
    int N_MAT;
    if (argc == 2) {
        N_MAT = atoi(argv[1]);
    } else {
        N_MAT = 4;
    }

    // #ifdef _OPENMP
    //     if (N_MAT > 64) {
    //         omp_set_dynamic(0);
    //         omp_set_num_threads(64);
    //         cout << "using 64 threads" << endl;
    //     }
    // #endif

#ifdef GPROF
    ProfilerStart("pg.log");
#endif

    for (int i = 0; i < 8; i++) {
        Mat A = Mat::rand_gen<int>(1024, 1024, 1, 16);

        auto start = std::chrono::high_resolution_clock::now();
        LU A_LU(std::move(A));

        bool succ;
        Mat Ainv = A_LU.inverse(succ);
        auto end = std::chrono::high_resolution_clock::now();
        auto interval =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
        cout << (double)interval * 1e-6 << "s" << endl;
    }

#ifdef GPROF
    ProfilerStop();
#endif
    // correctness test
    cout << endl << "==============================" << endl;
    cout << "correctness test:" << endl;
    Mat A(4, 4);

    A.at(0, 0) = 9;
    A.at(0, 1) = 6;
    A.at(0, 2) = 14;
    A.at(0, 3) = 12;
    A.at(1, 0) = 12;
    A.at(1, 1) = 6;
    A.at(1, 2) = 8;
    A.at(1, 3) = 11;
    A.at(2, 0) = 1;
    A.at(2, 1) = 3;
    A.at(2, 2) = 6;
    A.at(2, 3) = 6;
    A.at(3, 0) = 17;
    A.at(3, 1) = 3;
    A.at(3, 2) = 11;
    A.at(3, 3) = 4;
    cout << A << endl;

    auto start = std::chrono::high_resolution_clock::now();
    LU A_LU(std::move(A));

    bool succ;
    Mat Ainv = A_LU.inverse(succ);
    auto end = std::chrono::high_resolution_clock::now();
    auto interval =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    cout << (double)interval * 1e-6 << "s" << endl;
    cout << Ainv << endl;

    return 0;
}
