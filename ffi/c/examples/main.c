/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 TPT Solutions
 *
 * Fits KMeans to two blobs through the C ABI, checks every status code, and prints the labels.
 *
 * Build from ffi/c after `cargo build --release`:
 *
 *   cc -std=c99 -Wall -Wextra -Werror -Iinclude examples/main.c \
 *      -Ltarget/release -ltpt_ml_proven -o target/kmeans_example
 *   LD_LIBRARY_PATH=target/release ./target/kmeans_example     (Linux)
 *
 * On macOS set DYLD_LIBRARY_PATH instead; on Windows use the GNU toolchain and put the DLL
 * next to the executable.
 */

#include <stdint.h>
#include <stdio.h>

#include "tpt_ml_proven.h"

#define ROWS 6
#define COLS 2

static const double DATA[ROWS * COLS] = {
    0.0, 0.0, 0.1, 0.0, 0.0, 0.1,   /* the first blob */
    10.0, 10.0, 10.1, 10.0, 10.0, 10.1, /* the second blob */
};

int main(void) {
    uintptr_t labels[ROWS];
    enum tpml_TpmlStatus status;
    struct tpml_TpmlKMeans *model;
    int i;

    printf("tpt-ml-proven %s\n", tpml_version());

    /* n_clusters, max_iter, tol, seed, threads. A null handle means invalid settings. */
    model = tpml_kmeans_new(2, 300, 1e-4, 1, 1);
    if (model == NULL) {
        fprintf(stderr, "invalid KMeans settings\n");
        return 1;
    }

    status = tpml_kmeans_fit(model, DATA, ROWS, COLS);
    if (status != tpml_TpmlStatus_Ok) {
        fprintf(stderr, "fit failed (status %d): %s\n", (int)status, tpml_last_error_message());
        tpml_kmeans_free(model);
        return 1;
    }

    status = tpml_kmeans_labels(model, labels, ROWS);
    if (status != tpml_TpmlStatus_Ok) {
        fprintf(stderr, "labels failed (status %d): %s\n", (int)status, tpml_last_error_message());
        tpml_kmeans_free(model);
        return 1;
    }

    printf("labels:");
    for (i = 0; i < ROWS; i++) {
        printf(" %lu", (unsigned long)labels[i]);
    }
    printf("\n");

    tpml_kmeans_free(model);
    return 0;
}
