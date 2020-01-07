#ifndef __MAT_H__
#define __MAT_H__

#include <bits/stdc++.h>

#if __cplusplus < 201103L
#define nullptr NULL
#endif

typedef float data_t;

class RangeTrait;

struct Range;
struct RangeSingle;
struct RangeTo;
struct RangeFrom;
struct RangeFull;

struct Buffer;
struct Permuate;
struct Mat;
struct VecRefSlice;

std::ostream &operator<<(std::ostream &os, const Mat &mat);

void gauss_elim_step(Mat &mat, data_t diag, int i);
void gauss_elim_step_swap(Mat &mat, data_t diag, int i, int pivot);

void vec_axpy(data_t *yptr, data_t a, data_t *xptr, data_t p, int xstride,
              int ystride, int len);
void vec_ax(data_t *yptr, data_t a, data_t *xptr, int xstride, int ystride,
            int len);

void vec_axpy(VecRefSlice &yref, data_t a, VecRefSlice &xref, data_t p,
              int xstride, int ystride, int len);
void vec_ax(VecRefSlice &yref, data_t a, VecRefSlice &xref, int xstride,
            int ystride, int len);

struct VecRefSlice {
    data_t *ptr;
    int len;
    VecRefSlice(data_t *p, int l) : ptr(p), len(l) {}
    ~VecRefSlice() { ptr = nullptr; }
    VecRefSlice(VecRefSlice &&src) : ptr(src.ptr), len(src.len) {}

  private:
    VecRefSlice(VecRefSlice &src) = delete;
    VecRefSlice operator=(const VecRefSlice &src) = delete;
};

struct Buffer {
    bool is_slice;
    data_t *ptr;
    struct Shape {
        int r;
        int c;
        Shape() {}
        Shape(int _r, int _c) : r(_r), c(_c) {}
    } shape;
    struct Strides {
        int r = 1;
        int c;
        Strides(int _r, int _c) : r(_r), c(_c) {}
        Strides() {}
    } strides;

    // VecRefSlice ref_slice() { return VecRefSlice(ptr, serial_idx()); }

    inline int serial_idx(int ir, int ic) const {
        return ir * strides.r + ic * strides.c;
    }

    inline data_t *serial_at_ptr(int i) { return ptr + i; }
    inline const data_t *serial_at_ptr(int i) const { return ptr + i; }
    inline data_t *at_ptr(int ir, int ic) {
        return serial_at_ptr(serial_idx(ir, ic));
    }
    inline const data_t *at_ptr(int ir, int ic) const {
        return serial_at_ptr(serial_idx(ir, ic));
    }
    void swap(std::pair<int, int> rc1, std::pair<int, int> rc2) {
        int idx1 = serial_idx(rc1.first, rc1.second);
        int idx2 = serial_idx(rc2.first, rc2.second);

        std::swap(*serial_at_ptr(idx1), *serial_at_ptr(idx2));
    }

    Buffer() : ptr(nullptr), shape(0, 0), strides(0, 0), is_slice(false) {}

    Buffer(data_t *sptr, int r, int c, int rstride, int cstride)
        : ptr(sptr), shape(r, c), strides(rstride, cstride), is_slice(true) {}

    Buffer(int r, int c) : shape(r, c), strides(1, c), is_slice(false) {
        if (r * c != 0)
            ptr = new data_t[r * c];
    }
    Buffer(int r, int c, data_t v)
        : shape(r, c), strides(1, c), is_slice(false) {
        const int size = r * c;
        if (size != 0) {
#if __cplusplus > 201103L
            ptr = new data_t[size]{v};
#else
            ptr = new data_t[size];

            std::fill_n(ptr, size, v);
#endif
        }
    }

    inline void resize(int r, int c) {
        if (!is_slice && (shape.c != c) && (shape.r != r)) {
            delete[] ptr;
            ptr = new data_t[r * c];
            shape.r = r;
            shape.c = c;
            strides.c = c;
        }
    }

    Buffer(Buffer &src) = delete;
    Buffer operator=(const Buffer &src) = delete;
    Buffer(Buffer &&src)
        : shape(src.shape), strides(src.strides), ptr(src.ptr),
          is_slice(src.is_slice) {}

    static Buffer slice_with_strides(Buffer &src, int rstart, int cstart, int r,
                                     int c, int rstride, int cstride) {
        data_t *sptr = src.at_ptr(rstart, cstart);
        Buffer ret(sptr, r, c, rstride, cstride);

        return ret;
    }
    static Buffer slice(Buffer &src, int rstart, int cstart, int r, int c) {

        data_t *sptr = src.at_ptr(rstart, cstart);
        Buffer ret(sptr, r, c, src.strides.r, src.strides.c);
        return ret;
    }

    ~Buffer() {
        if (!is_slice && !ptr) {
            delete[] ptr;
        }
        ptr = nullptr;
    }
};

class RangeTrait {
  public:
    virtual int begin(int dim) = 0;
    virtual int end(int dim) = 0;
    virtual int size(int dim) = 0;
};

struct RangeSingle : public RangeTrait {
    int range;
    RangeSingle(int r) : range(r) {}
    virtual int begin(int dim) { return range; }
    virtual int end(int dim) { return range + 1; }
    virtual int size(int dim) { return 1; }
};

struct Range : public RangeTrait {
    int s;
    int e;
    Range(int start, int end) : s(start), e(end) {}
    virtual int begin(int dim) { return s; }
    virtual int end(int dim) { return e; }
    virtual int size(int dim) { return e - s; }
};

struct RangeTo : public RangeTrait {
    int e;
    RangeTo(int end) : e(end) {}
    virtual int begin(int dim) { return 0; }
    virtual int end(int dim) { return e; }
    virtual int size(int dim) { return e; }
};

struct RangeFrom : public RangeTrait {
    int s;
    RangeFrom(int start) : s(start) {}
    virtual int begin(int dim) { return s; }
    virtual int end(int dim) { return dim; }
    virtual int size(int dim) { return dim - s; }
};

struct RangeFull : public RangeTrait {
    RangeFull() {}
    virtual int begin(int dim) { return 0; }
    virtual int end(int dim) { return dim; }
    virtual int size(int dim) { return dim; }
};

// struct Range;
// struct RangeSingle;
// struct RangeTo;
// struct RangeFrom;

// struct Range {
//     int start;
//     int end;
//     int size;

//     static Range range(int s, int e) {
//         if (s != e)
//             return Range(s, e);
//         else
//             return range(s);
//     }
//     static Range range(int single) {
//         Range ret(single, single + 1);
//         ret.size = 1;
//         return ret;
//     }

//   private:
//     Range(int s, int e) : start(s), end(e) { size = e - s; }
// };

struct Mat {
    Buffer buf;
    Mat() : buf() {}
    Mat(int r, int c) : buf(r, c) {}
    Mat(int r, int c, data_t v) : buf(r, c, v) {}

    Mat(Mat &&src) : buf(std::move(src.buf)) {}
    Mat(Mat &src) = delete;
    Mat operator=(const Mat &src) = delete;

    static inline Mat zeros(int r, int c) { return Mat(r, c, data_t(0)); }
    static inline Mat ones(int r, int c) { return Mat(r, c, data_t(1)); }
    static inline Mat eye(int r, int c) {
        Mat ret = Mat::zeros(r, c);
        for (int i = 0; i < r; i++) {
            *ret.buf.at_ptr(i, i) = data_t(1);
        }
        return ret;
    }

    void copy_to(Mat &rhs) {
        int r = buf.shape.r;
        int c = buf.shape.c;
        rhs.resize(r, c);
        memcpy(rhs.buf.ptr, buf.ptr, r * c * sizeof(data_t));
    }

    inline void resize(int r, int c) { buf.resize(r, c); }

    template <class T>
    static inline auto rand_gen(int r, int c, int min, int max) ->
        typename std::enable_if<std::is_floating_point<T>::value, Mat>::type {
        Mat ret(r, c);

        std::random_device rd;  // get a random number
        std::mt19937 rng(rd()); // generator with seed

        std::uniform_real_distribution<T> distr(min, max);

        for (int i = 0; i < r * c; i++) {
            *ret.buf.serial_at_ptr(i) = (data_t)distr(rng);
        }
        return ret;
    }

    template <class T>
    static inline auto rand_gen(int r, int c, int min, int max) ->
        typename std::enable_if<std::is_integral<T>::value, Mat>::type {
        Mat ret(r, c);
        std::random_device rd;  // get a random number
        std::mt19937 rng(rd()); // generator with seed

        std::uniform_int_distribution<T> distr(min, max);

        for (int i = 0; i < r * c; i++) {
            *ret.buf.serial_at_ptr(i) = (data_t)distr(rng);
        }
        return ret;
    }

    void to_eye() {
        for (int i = 0; i < shape().r; i++) {
            for (int j = 0; j < shape().c; j++) {
                *buf.at_ptr(i, j) = data_t(0);
            }
        }

        for (int i = 0; i < shape().r; i++) {
            *buf.at_ptr(i, i) = data_t(1);
        }
    }

    Buffer::Shape shape() const { return buf.shape; }
    Buffer::Strides strides() const { return buf.strides; }

    bool is_square() { return (buf.shape.r == buf.shape.c); }

    void swap(std::pair<int, int> rc1, std::pair<int, int> rc2) {
        buf.swap(rc1, rc2);
    }

    Mat slice(int rstart, int cstart, int r, int c) {
        Buffer slice_buf = Buffer::slice(buf, rstart, cstart, r, c);
        return Mat(std::move(slice_buf));
    }

    Mat slice_range(RangeTrait &&rows, RangeTrait &&cols) {
        int r = buf.shape.r;
        int c = buf.shape.c;

        return slice(rows.begin(r), cols.begin(c), rows.size(r), cols.size(c));
    }

    int nrows() { return buf.shape.r; }
    int ncols() { return buf.shape.c; }

    // col vec mode
    data_t *v_at_ptr(int i) {
        int idx = i * buf.strides.r;
        return buf.serial_at_ptr(i);
    }

    // col vec mode
    const data_t &v_at(int i) const {
        int idx = i * buf.strides.r;
        return *buf.serial_at_ptr(i);
    }

    data_t &v_at(int i) {
        int idx = i * buf.strides.r;
        return *buf.serial_at_ptr(i);
    }

    data_t *at_ptr(int i, int j) { return buf.at_ptr(i, j); }
    const data_t *at_ptr(int i, int j) const { return buf.at_ptr(i, j); }

    data_t &at(int i, int j) { return *buf.at_ptr(i, j); }
    const data_t &at(int i, int j) const { return *buf.at_ptr(i, j); }

    int v_iamax() {
        data_t max = v_at(0), val;
        int idx = 0;
        for (int i = 0; i < nrows(); i++) {
            val = std::abs(v_at(i));
            if (val > max) {
                max = val;
                idx = i;
            }
        }
        return idx;
    }

    std::pair<int, int> mat_iamax() {
        data_t max = std::abs(at(0, 0)), val;
        auto ret = std::make_pair(0, 0);

        for (int j = 0; j < ncols(); j++) {
            for (int i = 0; i < nrows(); i++) {
                val = std::abs(at(i, j));

                if (val > max) {
                    max = val;
                    ret = std::make_pair(i, j);
                }
            }
        }

        return ret;
    }

    void swap_rows(int r1, int r2) {
        if (r1 != r2) {
            for (int i = 0; i < ncols(); i++) {
                // TODO use memcpy instead
                std::swap(at(r1, i), at(r2, i));
            }
        }
    }
    void swap_cols(int c1, int c2) {
        if (c1 != c2) {
            for (int i = 0; i < nrows(); i++) {
                // TODO maybe use memcpy instead
                std::swap(at(i, c1), at(i, c2));
            }
        }
    }

    Mat col(int c) {
        auto rows = buf.shape.r;
        Buffer tbuf = Buffer::slice(buf, 0, c, rows, 1);
        return Mat(std::move(tbuf));
    }

    Mat rows_range(RangeTrait &&rows) {
        RangeFull cols;
        return slice_range(std::move(rows), std::move(cols));
    }

    Mat cols_range(RangeTrait &&cols) {
        RangeFull rows;
        return slice_range(std::move(rows), std::move(cols));
    }

    void rows_range_pair(RangeTrait &&r1, RangeTrait &&r2, Mat &sub1,
                         Mat &sub2) {
        auto rows = buf.shape.r;
        auto cols = buf.shape.c;
        auto rstride = buf.strides.r;
        auto cstride = buf.strides.c;

        int m1start = r1.begin(rows);
        int m1end = r1.end(rows);
        int m1rows = r1.size(rows);

        int m2start = r2.begin(rows);
        int m2end = r2.end(rows);
        int m2rows = r2.size(rows);

        // overlaps and out-of-range check
        assert(m2start >= m1end || m1start >= m2end);
        assert(m2end <= rows);

        auto m1ptr = buf.at_ptr(m1start, 0);
        auto m2ptr = buf.at_ptr(m2start, 0);

        Buffer m1buf = Buffer(m1ptr, m1rows, cols, rstride, cstride);
        Buffer m2buf = Buffer(m2ptr, m2rows, cols, rstride, cstride);

        sub1.from_buf(std::move(m1buf));
        sub2.from_buf(std::move(m2buf));
    }

    void cols_range_pair(RangeTrait &&r1, RangeTrait &&r2, Mat &sub1,
                         Mat &sub2) {
        auto rows = buf.shape.r;
        auto cols = buf.shape.c;
        auto rstride = buf.strides.r;
        auto cstride = buf.strides.c;

        int m1start = r1.begin(cols);
        int m1end = r1.end(cols);
        int m1cols = r1.size(cols);

        int m2start = r2.begin(cols);
        int m2end = r2.end(cols);
        int m2cols = r2.size(cols);

        // printf("%d %d %d\n", m1start, m1end, m1cols);
        // printf("%d %d %d\n", m2start, m2end, m2cols);

        assert(m2start >= m1end || m1start >= m2end);
        assert(m2end <= cols);

        auto m1ptr = buf.at_ptr(0, m1start);
        auto m2ptr = buf.at_ptr(0, m2start);

        Buffer m1buf = Buffer(m1ptr, rows, m1cols, rstride, cstride);
        Buffer m2buf = Buffer(m2ptr, rows, m2cols, rstride, cstride);

        sub1.from_buf(std::move(m1buf));
        sub2.from_buf(std::move(m2buf));
    }

    void scale(data_t s) {
        for (int j = 0; j < ncols(); j++) {
            for (int i = 0; i < nrows(); i++) {
                at(i, j) *= s;
            }
        }
    }

    void axpy(data_t a, Mat &x, data_t p) {
        assert(nrows() == x.nrows());

        int yrstride = strides().r;
        int xrstride = x.strides().r;

        VecRefSlice yvref = as_vec_ref();
        VecRefSlice xvref = x.as_vec_ref();

        if (p != 0) {
            return vec_axpy(yvref, a, xvref, p, yrstride, xrstride, xvref.len);
        } else {
            return vec_ax(yvref, a, xvref, yrstride, xrstride, xvref.len);
        }
    }

    bool solve_lower_tri(Mat &b) {
        int C = b.ncols();

#ifdef _OPENMP
        bool ret = true;
#pragma omp parallel for
        for (int i = 0; i < C; i++) {
            auto bcol = b.col(i);
            if (!solve_lower_tri_vec(bcol)) {
                ret = false;
            }
        }
        return ret;
#else
        for (int i = 0; i < C; i++) {
            auto bcol = b.col(i);
            if (!solve_lower_tri_vec(bcol)) {
                return false;
            }
        }

        return true;
#endif
    }

    bool solve_lower_tri_vec(Mat &b) {
        int R = nrows();

        for (int i = 0; i < R; i++) {
            data_t diag = at(i, i);

            if (diag == 0) {
                return false;
            }

            data_t coeff = b.at(i, 0) /= diag;
            b.at(i, 0) = coeff;

            auto x = slice_range(RangeFrom(i + 1), RangeSingle(i));

            b.rows_range(RangeFrom(i + 1)).axpy(-coeff, x, data_t(1));
        }
        return true;
    }

    bool solve_lower_tri_with_diag(Mat &b, data_t diag) {
        if (diag == 0) {
            return false;
        }

        int R = nrows();
        int C = b.ncols();

#pragma omp parallel for
        for (int j = 0; j < C; j++) {
            auto bcol = b.col(j);
            // -1
            for (int i = 0; i < R - 1; i++) {
                data_t coeff = bcol.at(i, 0) / diag;
                auto x = slice_range(RangeFrom(i + 1), RangeSingle(i));
                bcol.rows_range(RangeFrom(i + 1)).axpy(-coeff, x, data_t(1));
            }
        }

        return true;
    }

    bool solve_upper_tri(Mat &b) {
        int C = ncols();

#ifdef _OPENMP
        bool ret = true;
#pragma omp parallel for
        for (int i = 0; i < C; i++) {
            auto bcol = b.col(i);
            if (!solve_upper_tri_vec(bcol)) {
                ret = false;
            }
        }
        return ret;
#else
        for (int i = 0; i < C; i++) {
            auto bcol = b.col(i);
            if (!solve_upper_tri_vec(bcol)) {
                return false;
            }
        }
        return true;
#endif
    }
    bool solve_upper_tri_vec(Mat &b) {
        int R = nrows();
        for (int i = R - 1; i >= 0; i--) {
            data_t diag = at(i, i);
            if (diag == 0) {
                return false;
            }

            data_t coeff = b.at(i, 0) / diag;
            b.at(i, 0) = coeff;

            auto x = slice_range(RangeTo(i), RangeSingle(i));
            b.rows_range(RangeTo(i)).axpy(-coeff, x, data_t(1));
        }
        return true;
    }

  private:
    Mat(Buffer &&b) : buf(std::move(b)) {}
    void from_buf(Buffer &&b) {
        buf.ptr = b.ptr;
        buf.shape.r = b.shape.r;
        buf.shape.c = b.shape.c;
        buf.strides.r = b.strides.r;
        buf.strides.c = b.strides.c;
        buf.is_slice = true;
    }

    VecRefSlice as_vec_ref() {
        data_t *ptr = buf.ptr;
        int size = buf.serial_idx(buf.shape.r - 1, buf.shape.c - 1);
        if ((buf.shape.r != 0) && (buf.shape.c != 0)) {
            return VecRefSlice(ptr, size + 1);
        } else {
            return VecRefSlice(ptr, 0);
        }
    }
};

struct Permuate {
    int n;
    std::vector<std::pair<int, int>> pivot;
    Permuate() {}
    Permuate(int len) : n(0), pivot(len, std::pair<int, int>()) {}
    void add_permuate(int lhs, int rhs) {
        if (lhs != rhs) {
            // assume at most pivot.size() permuate
            pivot[n] = std::make_pair(lhs, rhs);
            n++;
        }
    }
    void permuate_rows(Mat &rhs) {
        for (int i = 0; i < n; i++) {
            auto p = pivot[i];
            rhs.swap_rows(p.first, p.second);
        }
    }
};

struct LU {
    Mat lu;
    Permuate p;

    LU(Mat &&src) : lu(std::move(src)) {
        auto shape = lu.buf.shape;
        int min_side = std::min(shape.r, shape.c);
        p = Permuate(min_side);

        for (int i = 0; i < min_side; i++) {
            int pivot =
                lu.slice_range(RangeFrom(i), RangeSingle(i)).v_iamax() + i;
            data_t diag = lu.at(pivot, i);

            if (diag == 0) {
                continue;
            }
            if (pivot != i) {
                p.add_permuate(i, pivot);
                // cols range
                lu.cols_range(RangeTo(i)).swap_rows(i, pivot);
                gauss_elim_step_swap(lu, diag, i, pivot);
            } else {
                gauss_elim_step(lu, diag, i);
            }
        }
    }

    bool solve(Mat &b) {
        assert(lu.nrows() == b.nrows());
        assert(lu.is_square());

        p.permuate_rows(b);
        lu.solve_lower_tri_with_diag(b, data_t(1));

        return lu.solve_upper_tri(b);
    }

    Mat inverse(bool &succ) {
        assert(lu.is_square());
        auto shape = lu.shape();
        auto inv = Mat::eye(shape.r, shape.c);
        succ = solve(inv);

        return inv;
    }

    void inverse_to(Mat &inv, bool &succ) {
        assert(lu.is_square());
        assert(inv.shape().r == lu.shape().r);
        assert(inv.shape().c == lu.shape().c);

        inv.to_eye();
        succ = solve(inv);
    }
};

#endif // __MAT_H__