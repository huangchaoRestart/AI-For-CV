/*author:huangchao
key points:
1. 使用优先级队列决定下次走哪一个step
2. 自定义优先级队列的比较函数
*/

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <sqlite3.h>
#include <complex>
#include <queue>
using namespace std;


vector<vector<int>> maze =
{
	{1, 5, 5, 5, 1},
    {1, 2, 1, 2, 1},
    {1, 1, 1, 1, 1},
    {3, 4, 3, 4, 1},
	{3, 4, 5, 3, 1}
};

vector<vector<int>> visited(maze.size(),vector<int>(maze[0].size(),-1));

struct cmp
{
    bool operator()(pair<int,int> &a,pair<int,int> &b)
    {
        //因为优先出列判定为!cmp，所以反向定义实现最小值优先
        return visited[a.first][a.second]>visited[b.first][b.second];
    }
};

void print_out(priority_queue<pair<int,int>,vector<pair<int,int>>,cmp> q){
	while(!q.empty()){
		pair<int,int> qe=q.top();
		q.pop();
		cout<<qe.first<<" "<<qe.second<<" "<<visited[qe.first][qe.second]<<endl;
	}
};

int getShortestPath(vector<vector<int>>& maze, pair<int, int> start, pair<int, int> end){
// never forget to check corner cases [or ask your interviewer to make sure your inputs arevalid]
	int row=maze.size();
	if(0==row) return -1;
	int col=maze[0].size();
	if(0==col) return -1;
	//valid start and end coord
	if(start.first<0 || start.first>=row || start.second<0 || start.second>=col
		|| end.first<0 || end.first>=row || end.second<0 || end.second>=col){
		return -1;	
	}
	
	vector<pair<int,int>> dirs={{1,0},{-1,0},{0,-1},{0,1}};
	visited[start.first][start.second]=maze[start.first][start.second];
	priority_queue<pair<int,int>,vector<pair<int,int>>,cmp> q;
	//priority_queue<pair<int,int>> q;
	q.push(start);
	
	while(!q.empty()){
		pair<int,int> cur=q.top();
		print_out(q);
		cout<<"top:"<<cur.first<<" "<<cur.second<<endl;
		cout<<endl;
		q.pop();
		if(cur.first==end.first && cur.second==end.second){
			return visited[cur.first][cur.second];
		}
		for(auto dir:dirs){
			int new_r=cur.first+dir.first;
			int new_c=cur.second+dir.second;
			if(new_r<0 || new_r>=row || new_c<0 || new_c>=col || visited[new_r][new_c]!=-1 || maze[new_r][new_c]==0){
				continue;
			}
			visited[new_r][new_c] = visited[cur.first][cur.second] +maze[new_r][new_c];
			q.push(pair<int,int>{new_r,new_c});
		}
		cout<<endl;
	}
}

int main() {

	pair<int, int> start = { 0, 0 };
	pair<int, int> end = {1, 1};
	int shortestPath = 0;
	shortestPath = getShortestPath(maze, start, end);
	std::cout << "shortest path length: " << shortestPath << endl;
	
	return 0;
}
