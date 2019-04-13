// convolution.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>

using namespace std;

vector<vector<int>> medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way)
{
	///////////ergodic W*H and merge sort m*n, so total time complexity is W*H*m*n*log(m*n)
	int H = img.size(), W = img[0].size(),m=kernel.size(),n=kernel[0].size();
	int padding_m = m / 2, padding_n = n / 2;
	vector<int> conv_element;
	vector<vector<int>> result(img);
	for (int p = 0; p < H; ++p) {
		for (int q = 0; q < W; ++q) {
			//choose convolution elements for kernel slide
			conv_element.clear();
			if (padding_way == "REPLICA") {
				for (int j = -padding_m; j <= padding_m; ++j) {
					for (int k = -padding_n; k <= padding_n; ++k) {
						conv_element.push_back(img[min(max(p + j, 0),H-1)][min(max(q + k, 0),W-1)]);
						}
				}
			}else if (padding_way == "ZERO") {
				for (int j = -padding_m; j <= padding_m; ++j) {
					for (int k = -padding_n; k <= padding_n; ++k) {
						if (p + j < 0 || q + k < 0 || p+j>=H ||q+k>=W) {
							conv_element.push_back(0);
						}
						else {
							conv_element.push_back(img[std::max(p + j, 0)][std::max(q + k, 0)]);
						}
					}
				}
			}
			//merge sort with time complexity m*n*log(m*n)
			std::sort(conv_element.begin(), conv_element.end());
			result[p][q] = conv_element[conv_element.size() / 2];
		}
	}
	return result;

}

int main()
{
	vector<vector<int>> image{ {1,2,3,4,5},{2,3,4,5,7},{2,3,5,7,3},{2,3,5,2,3} };
	vector<vector<int>> kernel{ {1,1,1},{1,1,1},{1,1,1} };
	vector<vector<int>> result = medianBlur(image, kernel, "REPLICA");
	for (int i = 0; i < result.size(); ++i) {
		for (int j = 0; j < result[0].size(); ++j)
			cout << result[i][j] << " ";
		cout << endl;
	}
	int input;
	cin >>input;
    return 0;
}

