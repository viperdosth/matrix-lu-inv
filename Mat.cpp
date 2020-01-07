#include "Mat.hpp"

std::ostream &operator<<(std::ostream &os, const Mat &mat) {
    const int r = mat.shape().r;
    const int c = mat.shape().c;
    // os << "shape " << r << " " << c << std::endl;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            os << mat.at(i, j) << " ";
        }
        os << std::endl;
    }

    return os;
}

void gauss_elim_step(Mat &mat, data_t diag, int i) {
    Mat submat = mat.slice_range(RangeFrom(i), RangeFrom(i));

    data_t inv_diag = data_t(1) / diag;

    Mat coes;
    Mat smat;
    submat.cols_range_pair(RangeSingle(0), RangeFrom(1), coes, smat);

    Mat coeffs = coes.rows_range(RangeFrom(1));
    coeffs.scale(inv_diag);

    Mat rpivot;
    Mat down;
    smat.rows_range_pair(RangeSingle(0), RangeFrom(1), rpivot, down);

    int rpivot_cols = rpivot.ncols();
    for (int k = 0; k < rpivot_cols; k++) {
        down.col(k).axpy(-rpivot.at(0, k), coeffs, 1);
    }
}

void gauss_elim_step_swap(Mat &mat, data_t diag, int i, int pivot) {
    int piv = pivot - i;
    Mat submat = mat.slice_range(RangeFrom(i), RangeFrom(i));
    data_t inv_diag = data_t(1) / diag;

    Mat coes;
    Mat smat;
    submat.cols_range_pair(RangeSingle(0), RangeFrom(1), coes, smat);

    coes.swap(std::make_pair(0, 0), std::make_pair(piv, 0));
    Mat coeffs = coes.rows_range(RangeFrom(1));
    coeffs.scale(inv_diag);

    Mat rpivot;
    Mat down;
    smat.rows_range_pair(RangeSingle(0), RangeFrom(1), rpivot, down);

    int rpivot_cols = rpivot.ncols();
    for (int k = 0; k < rpivot_cols; k++) {
        std::swap(rpivot.at(0, k), down.at(piv - 1, k));
        auto tmp = down.col(k);
        down.col(k).axpy(-rpivot.at(0, k), coeffs, 1);
    }
}

void vec_axpy(data_t *yptr, data_t a, data_t *xptr, data_t p, int ystride,
              int xstride, int len) {
    for (int i = 0; i < len; i++) {
        data_t *y = yptr + i * ystride;
        *y = a * (*(xptr + i * xstride)) + p * (*y);
    }
}

void vec_ax(data_t *yptr, data_t a, data_t *xptr, int xstride, int ystride,
            int len) {
    for (int i = 0; i < len; i++) {
        data_t *y = yptr + i * ystride;
        *y = a * (*(xptr + i * xstride));
    }
}

void vec_axpy(VecRefSlice &yref, data_t a, VecRefSlice &xref, data_t p,
              int xstride, int ystride, int len) {
    data_t *yptr = yref.ptr, *xptr = xref.ptr;

    // for (int i = 0; i < len; i++) {
    //     data_t *y = yptr + i * ystride;
    //     *y = a * (*(xptr + i * xstride)) + p * (*y);
    // }

    int len_div_u = len / 4;
    int ulen = len_div_u * 4;
    int i;

    for (i = 0; i < len_div_u; i++) {
        data_t *y0 = yptr + i * ystride;
        data_t *y1 = yptr + (i + 1) * ystride;
        data_t *y2 = yptr + (i + 2) * ystride;
        data_t *y3 = yptr + (i + 3) * ystride;

        data_t y0v = *(yptr + i * ystride);
        data_t y1v = *(yptr + (i + 1) * ystride);
        data_t y2v = *(yptr + (i + 2) * ystride);
        data_t y3v = *(yptr + (i + 3) * ystride);

        data_t x0 = *(xptr + i * xstride);
        data_t x1 = *(xptr + (i + 1) * xstride);
        data_t x2 = *(xptr + (i + 2) * xstride);
        data_t x3 = *(xptr + (i + 3) * xstride);

        *y0 = a * x0 + p * y0v;
        *y1 = a * x1 + p * y1v;
        *y2 = a * x2 + p * y2v;
        *y3 = a * x3 + p * y3v;
    }

    if (len - ulen) {
        for (i = ulen; i < len; i++) {
            data_t *y = yptr + i * ystride;
            data_t x = *(xptr + i * xstride);
            *y = a * x + p * (*y);
        }
    }
}
void vec_ax(VecRefSlice &yref, data_t a, VecRefSlice &xref, int xstride,
            int ystride, int len) {
    data_t *yptr = yref.ptr, *xptr = xref.ptr;

    // for (int i = 0; i < len; i++) {
    //     data_t *y = yptr + i * ystride;
    //     *y = a * (*(xptr + i * xstride));
    // }

    int len_div_u = len / 4;
    int ulen = len_div_u * 4;
    int i;

    for (int i = 0; i < len_div_u; i++) {
        data_t *y0 = yptr + i * ystride;
        data_t *y1 = yptr + (i + 1) * ystride;
        data_t *y2 = yptr + (i + 2) * ystride;
        data_t *y3 = yptr + (i + 3) * ystride;

        data_t x0 = *(xptr + i * xstride);
        data_t x1 = *(xptr + (i + 1) * xstride);
        data_t x2 = *(xptr + (i + 2) * xstride);
        data_t x3 = *(xptr + (i + 3) * xstride);

        *y0 = a * x0;
        *y1 = a * x1;
        *y2 = a * x2;
        *y3 = a * x3;
    }

    if (len - ulen) {
        for (i = ulen; i < len; i++) {
            data_t *y = yptr + i * ystride;
            data_t x = *(xptr + i * xstride);
            *y = a * x;
        }
    }
}