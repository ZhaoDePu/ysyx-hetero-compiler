#include "c11_qwen_ref.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) {
        fputs("usage: c11_qwen_rope_fixture WEIGHTS TOKEN_IDS OUTPUT [VECTORS]\n", stderr);
        return 2;
    }
    qwen_reference *model = qwen_reference_open(argv[1]);
    FILE *input = fopen(argv[2], "r"), *out = fopen(argv[3], "wb");
    FILE *vec = argc == 5 ? fopen(argv[4], "wb") : NULL;
    int rc = 1;
    if (model && input && out && (argc == 4 || vec)) {
        qwen_layer0_rope fixture;
        qwen_layer0_vectors vectors;
        qwen_reference_capture_rope(model, &fixture);
        if (vec)
            qwen_reference_capture_vectors(model, &vectors);
        rc = 0;
        for (unsigned pos = 0; pos < 10; ++pos) {
            unsigned token;
            if (fscanf(input, "%u", &token) != 1 || !qwen_reference_step(model, token) ||
                (fwrite(&fixture, sizeof(fixture), 1, out) != 1 ||
                 (vec && fwrite(&vectors, sizeof(vectors), 1, vec) != 1))) {
                rc = 1;
                break;
            }
        }
    }
    if (input)
        fclose(input);
    if (out && fclose(out))
        rc = 1;
    if (vec && fclose(vec))
        rc = 1;
    qwen_reference_close(model);
    return rc;
}
