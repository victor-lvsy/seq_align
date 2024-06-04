#include "nw.h"


void nw(const std::string &seq1, const std::string &seq2, int match, int mismatch, int gap) {

    int n = seq1.length();
    int m = seq2.length();
    std::vector<std::vector<int>> score(n + 1, std::vector<int>(m + 1, 0));

    // Initialization
    for (int i = 0; i <= n; ++i)
        score[i][0] = i * gap;

    for (int j = 0; j <= m; ++j)
        score[0][j] = j * gap;

    // Matrix filling
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int match_score = (seq1[i - 1] == seq2[j - 1]) ? match : mismatch;
            score[i][j] = std::max({
                score[i - 1][j - 1] + match_score,
                score[i - 1][j] + gap,
                score[i][j - 1] + gap
            });
        }
    }

    // Print the score matrix for debugging
    // for (int i = 0; i <= n; ++i) {
    //     for (int j = 0; j <= m; ++j) {
    //         std::cout << score[i][j] << " ";
    //     }
    //         std::cout << std::endl;
    // }   

    // Backtracking
    int i = n;
    int j = m;
    std::string aligned_seq1, aligned_seq2;

    while (i > 0 && j > 0) {
        if (score[i][j] == score[i - 1][j - 1] + (seq1[i - 1] == seq2[j - 1] ? match : mismatch)) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --i;
            --j;
        } else if (score[i][j] == score[i - 1][j] + gap) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = "-" + aligned_seq2;
            --i;
        } else {
            aligned_seq1 = "-" + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --j;
        }
    }

    while (i > 0) {
        aligned_seq1 = seq1[i - 1] + aligned_seq1;
        aligned_seq2 = "-" + aligned_seq2;
        --i;
    }

    while (j > 0) {
        aligned_seq1 = "-" + aligned_seq1;
        aligned_seq2 = seq2[j - 1] + aligned_seq2;
        --j;
    }

    std::cout << "Aligned Sequence 1: " << aligned_seq1 << std::endl;
    std::cout << "Aligned Sequence 2: " << aligned_seq2 << std::endl;

}