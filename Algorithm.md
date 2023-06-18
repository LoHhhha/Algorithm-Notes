[TOC]
# **图论算法**

## 1. 最短路径

### A. Dijkstra(单源问题)

* 细节

        每次贪心地将最短的、可以拓展点的边并入集合，最终获得最短的路径。
        每次加入点之后更新最小距离，寻找到下一个最短路径的点加入集合。（不保证得到的是一颗树，但保证点到点最小）

* 代码实现

    ```C++
    vector<int> Dijkstra(vector<vector<pair<int,int>>>&edges,int k){
        //传入k是起始点
        //初始化为不可达
        int n=edges.size();
        vector<int>dicts(n,0x3f3f3f3f);
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>qu;
        //存储当前距离，当前节点
        qu.push({0,k});
        dicts[k]=0;
        while(!qu.empty()){
            auto [d,u]=qu.top();
            qu.pop();
            if(dicts[u]<d)continue;
            for(auto &[v,w]:edges[u]){
                int d1=d+w;
                if(dicts[v]>d1){
                    qu.push({d1,v});
                    dicts[v]=d1;
                }
            }
        }
        return dicts;
    }
    ```

* 封装

    ```C++
    auto Dijkstra=[&](vector<vector<pair<int,int>>>&edges,int k) -> vector<int> {
        int n=edges.size();
        vector<int>dicts(n,0x3f3f3f3f);
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>qu;
        qu.push({0,k});
        dicts[k]=0;
        while(!qu.empty()){
            auto [d,u]=qu.top();
            qu.pop();
            if(dicts[u]<d)continue;
            for(auto &[v,w]:edges[u]){
                int d1=d+w;
                if(dicts[v]>d1){
                    qu.push({d1,v});
                    dicts[v]=d1;
                }
            }
        }
        return dicts;
    };
    ```

* 分析

    * 时间复杂度： O(eloge)
    * 空间复杂度： O(e)
    
### B. BellmanFord(单源问题)

适用于负边

* 分析

        要路径最小，一条路径长度最多为n-1，否则形成环不可能最短，而对于每一条边都遍历n-1次，看在什么时候加入最优，以达成找到最短路径的目的。

* 代码实现

    ```c++
    vector<int> BellmanFord(vector<vector<int>>&edges,int n,int k){
        //边信息edges:edges[i]={u,v,w} 顶点数n 出发节点k
        vector<int>dist(n,0x3f3f3f3f);
        dist[k]=0;
        //最多经过n-1次
        for(int i=0;i<n-1;i++){
            for(auto &edge:edges){
                if(dist[edge[0]]+edge[2]<dist[edge[1]]){
                    dist[edge[1]] = dist[edge[0]] + edge[2];
                }
            }
        }
        return dist;
    }
    ```

* 封装

    ```c++
    auto BellmanFord=[&](vector<vector<int>>&edges,int n,int k) -> vector<int> {
        vector<int>dist(n,0x3f3f3f3f);
        dist[k]=0;
        for(int i=0;i<n-1;i++){
            for(auto &edge:edges){
                if(dist[edge[0]]+edge[2]<dist[edge[1]]){
                    dist[edge[1]] = dist[edge[0]] + edge[2];
                }
            }
        }
        return dist;
    };
    ```

* 分析

    * 时间复杂度： O(ne)

    * 空间复杂度： O(n)

### C. Floyd(多源问题)

* 细节

        利用动态规划的思想，将各点之间的距离求出。转移方程为dp[i][j]=min(dp[i][k]+dp[k][j],dp[i][j])，其中k∈集合内的点，每次先确定当前的中间点k，将中间点一个一个加进去选取最优，而不是先枚举ij后选择k，这样会导致在某些时候距离不正确。

* 代码实现

    ```c++
    vector<vector<int>> Floyd(vector<vector<pair<int,int>>>&edges){
        int n=edges.size();
        vector<vector<int>>dp(n,vector<int>(n,0x3f3f3f3f));
        //这一步可有可无，dp[i][i]的状态不会影响到最后其余点，但不初始化不要使用dp[i][i]
        for(int i=0;i<n;i++){
            dp[i][i]=0;
        }
        for(int i=0;i<n;i++){
            for(auto &[u,w]:edges[i]){
                dp[i][u]=w;
            }
        }
        //注意是先枚举k，再枚举ij
        for(int k=0;k<n;k++){
            for(int i=0;i<n;i++){
                for(int j=0;j<n;j++){
                    dp[i][j]=min(dp[i][j],dp[i][k]+dp[k][j]);
                }
            }
        }
        return dp;
    }
    ```

* 封装

    ```c++
    auto Floyd=[&](vector<vector<pair<int,int>>>&edges) -> vector<vector<int>> {
        int n=edges.size();
        vector<vector<int>>dp(n,vector<int>(n,0x3f3f3f3f));
        for(int i=0;i<n;i++){
            dp[i][i]=0;
        }
        for(int i=0;i<n;i++){
            for(auto &[u,w]:edges[i]){
                dp[i][u]=w;
            }
        }
        for(int k=0;k<n;k++){
            for(int i=0;i<n;i++){
                for(int j=0;j<n;j++){
                    dp[i][j]=min(dp[i][j],dp[i][k]+dp[k][j]);
                }
            }
        }
        return dp;
    };
    ```


* 分析

    * 时间复杂度： O($n^3$)

    * 空间复杂度： O($n^2$)

### D. BFS/DFS解决固定边权问题

用于边权固定的图。

* BFS

    * 代码实现

        ```C++
        // 以k节点开始，以及预处理好邻接表edges，顶点数n
        vector<int>dicts(n);
        vector<bool>visited(n);
        queue<int>qu;
        qu.push(k);
        visited[k]=true;
        int depth=1;
        while(!qu.empty()){
            int nq=qu.size();
            while(nq--){
                int u=qu.front();
                qu.pop();
                for(auto &v:edges[u]){
                    if(!visited[v]){
                        dicts[v]=depth;
                        qu.push(v);
                        visited[v]=true;
                    }
                }
            }
            depth++;
        }
        ```

* BFS维护最短路径

    维护一个pre数组，存储当前节点的上一个节点。

    ```C++
    // 以k节点开始，以及预处理好邻接表edges，顶点数n
    vector<int>dicts(n);
    vector<bool>visited(n);
    vector<int>pre(n);
    iota(pre.begin(),pre.end(),0);
    queue<int>qu;
    qu.push(k);
    visited[k]=true;
    while(!qu.empty()){
        int nq=qu.size();
        while(nq--){
            int u=qu.front();
            qu.pop();
            for(auto &v:edges[u]){
                if(!visited[v]){
                    pre[v]=u;
                    qu.push(v);
                    visited[v]=true;
                }
            }
        }
    }
    ```
    ```C++
    // 起点k到任一点e的路径可以通过以下方法求得，保证k、e互达。
    ptr=e;
    vector<int>round;
    while(pre[ptr]!=ptr){
        round.push_back(ptr);
        ptr=pre[ptr];
    }
    round.push_back(ptr);
    reverse(round.begin(),round.end());
    ```

* 01-BFS最短路

    相对于DJ更小的常数。

    * 细节

        解决只存在边权为0/x的图的最短路。最优的就是先使用0边权的边，其次再使用x边权的边。对应到实现上就是将0边权的放在队列前，将x边权的放在队列后。

    * 代码示例

        空网格（$grid[r][c]==0$）前往无代价，有障碍网格（$grid[r][c]==1$）代价为1。
        ```C++
        int minimumObstacles(vector<vector<int>>& grid) {
            int n=grid.size(),m=grid[0].size();
            vector<vector<int>>d(n,vector<int>(m,INT_MAX));
            d[0][0]=0;
            deque<pair<int,int>>q;
            int dirs[4][2]={{0,1},{1,0},{0,-1},{-1,0}};
            q.push_back({0,0});
            while(!q.empty()){
                auto [r,c]=q.front();
                q.pop_front();
                for(auto &[dr,dc]:dirs){
                    int r0=r+dr,c0=c+dc;
                    if(r0>=0&&c0>=0&&r0<n&&c0<m&&d[r0][c0]>d[r][c]+grid[r0][c0]){
                        d[r0][c0]=d[r][c]+grid[r0][c0];
                        grid[r0][c0]?q.push_back({r0,c0}):q.push_front({r0,c0});
                    }
                }
            }
            return d[n-1][m-1];
        }
        ```

## 2. 拓扑排序

找到环、非环元素。

* 细节

        不断将入度为0的节点去除，并减少其指向的节点的度，再次检查是否有入度为0的节点。最终获得的就是图中的环。

* 代码实现

    * BFS

        ```c++
        vector<bool> Topologic(vector<vector<int>>&edges,int n){
            //p::edges[i]:{ui,vi}  n:节点数
            //r::true->非环元素 false->环元素
            vector<int>indeg(n);
            vector<bool>visited(n);
            vector<vector<int>>next(n);
            for(auto &edge:edges){
                next[edge[0]].push_back(edge[1]);
                indeg[edge[1]]++;
            }
            queue<int>qu;
            for(int i=0;i<n;i++){
                if(indeg[i]==0){
                    qu.push(i);
                }
            }
            while(!qu.empty()){
                int cur=qu.front();
                qu.pop();
                visited[cur]=true;
                for(auto &v:next[cur]){
                    indeg[v]--;
                    if(indeg[v]==0){
                        qu.push(v);
                    }
                }
            }
            return visited;
        }
        ```
    * DFS

        ```c++
        vector<bool> Topologic(vector<vector<int>>&edges,int n){
            //edges[i]:{ui,vi}  n:节点数
            vector<int>indeg(n);
            vector<bool>visited(n);
            vector<vector<int>>next(n);
            for(auto &edge:edges){
                next[edge[0]].push_back(edge[1]);
                indeg[edge[1]]++;
            }
            function<void(int)> dfs=[&](int cur){
                visited[cur]=true;
                for(auto &v:next[cur]){
                    indeg[v]--;
                    if(!indeg[v]){
                        dfs(v);
                    }
                }
            };
            for(int i=0;i<n;i++){
                if(!indeg[i]&&!visited[i]){
                    dfs(i);
                }
            }
            return visited;
        }
        ```

* 分析

    * 时间复杂度： O(n+e)

    * 空间复杂度： O(n+e)

## 3. 二部图

### A. 最大匹配问题

* 细节

        匈牙利算法。
        在匹配不上的时候尝试改变配对，直到无法改变。
        最大匹配数等于最小点覆盖数（去掉最少的点清除所有的边）。

* 代码实现

    ```C++
    auto getMaxMatch=[&](vector<vector<int>>&next){
        int _n=next.size();

        queue<int>left;
        queue<int>qu;
        vector<bool>visited(_n);
        /* 获取左端点 */
        for(int i=0;i<_n;i++){
            if(!visited[i]){
                bool isleft=1;
                qu.push(i);
                visited[i]=true;
                while(!qu.empty()){
                    int qn=qu.size();
                    while(qn--){
                        int cur=qu.front();
                        qu.pop();
                        if(isleft)left.push(cur);
                        for(auto &v:next[cur]){
                            if(!visited[v]){
                                visited[v]=true;
                                qu.push(v);
                            }
                        }
                    }
                    isleft=!isleft;
                }
            }
        }

        /* 尝试推测 */
        vector<bool>used(_n);
        vector<int>partner(_n,-1);
        function<bool(int)> match=[&](int cur){
            for(auto &v:next[cur]){
                if(!used[v]){
                    used[v]=true;
                    if(partner[v]==-1||match(partner[v])){
                        partner[v]=cur;
                        return true;
                    }
                }
            }
            return false;
        };

        int cnt=0;
        while(!left.empty()){
            int cur=left.front();
            left.pop();
            used.assign(_n,false);
            if(match(cur)){
                cnt++;
            }
        }
        return cnt;
    };
    ```

* 分析

    * 时间复杂度：O($n^2m^2$)
    
    * 空间复杂度：O(nm)

## 4. 欧拉图

### A. 寻找欧拉通路/欧拉回路

* Hierholzer算法

    * 细节

            流程如下：
            1. 从起点出发，进行深度优先搜索。起点为任意（欧拉图）/为出度比入度大1的点。
            2. 每次沿着某条边从某个顶点移动到另外一个顶点的时候，都需要删除这条边。
            3. 如果没有可移动的路径，则将所在节点加入到栈中，并返回。
            4. 依次取出栈元素，就是一条欧拉路径。
            先遍历再放入是本算法的关键，能走通的节点总比不难走通的节点后入栈（反转后先经过）。
    
    * 代码实现

        ```C++
        vector<int> findItinerary(int n,vector<vector<int>>& edges) {
            vector<vector<int>>next(n);
            vector<pair<int,int>>degree(n);
            for(auto &edge:edges){
                next[edge[0]].push_back(edge[1]);
                degree[edge[0]].first++;
                degree[edge[1]].second++;
            }
            int start_place=0;
            for(auto &[place,d]:degree){
                if(d.first>d.second){
                    start_place=place;
                    break;
                }
            }
            vector<int>ret;
            function<void(int)>dfs=[&](int u){
                while(!next[u].empty()){
                    int v=next[u].back();
                    next[u].pop_back();
                    dfs(v);
                }
                ret.push_back(u);
            };
            dfs(start_place);
            reverse(ret.begin(),ret.end());
            return ret;
        }
        ```

    * 分析

        * 时间复杂度：O(n)
        
        * 空间复杂度：O(n)

## 5. 树

* 二叉树

    ```c++
    //Definition for a binary tree node.
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };
    ```

* N叉树

    ```c++
    //Definition for a Node.
    class Node {
    public:
        int val;
        vector<Node*> children;
    
        Node() {}
    
        Node(int _val) {
            val = _val;
        }
    
        Node(int _val, vector<Node*> _children) {
            val = _val;
            children = _children;
        }
    };
    ```

### A. 树的遍历

#### i. 先序遍历

* 细节

        二叉树：根->左子树->右子树
        N叉树：根->子树1->...->子树n

* 代码实现

    * DFS

        * 二叉树

            ```c++
            void Preorder(TreeNode* root){
                if(!root)return;
                /* do something */
                if(root->left)Preorder(root->left);
                if(root->right)Preorder(root->right);
            }
            ```

        * N叉树

            ```c++
            void Preorder(Node* root){
                if(!root)return;
                /* do something */
                for(auto &child:root->children){
                    Preorder(child);
                }
            }
            ```

    * BFS

        * 二叉树

            ```c++
            void Preorder(TreeNode* root){
                stack<TreeNode*>stk;
                TreeNode* ptr=root;
                while(ptr||!stk.empty()){
                    if(ptr){
                        /* do something */
                        stk.push(ptr);
                        ptr=ptr->left;
                    }
                    else{
                        ptr=stk.top();
                        stk.pop();
                        ptr=ptr->right;
                    }
                }
            }
            ```

        * N叉树

            ```c++
            vector<int> Preorder(Node* root) {
                stack<Node *> stk;
                if(root)st.push(root);
                while(!st.empty()) {
                    Node * node = st.top();
                    st.pop();
                    /* do something */
                    for (auto it = node->children.rbegin(); it != node->children.rend(); it++) {
                        st.push(*it);
                    }
                }
                return res;
            }
            ```

#### ii. 中序遍历

* 细节

        二叉树：左子树->根->右子树
        N叉树：没意义，无法确定“中”

* 代码实现

    * DFS

        ```c++
        void Inorder(TreeNode* root){
            if(!root)return;
            if(root->left)Inorder(root->left);
            /* do something */
            if(root->right)Inorder(root->right);
        }
        ```
    
    * BFS

        ```c++
        void Inorder(TreeNode* root){
            stack<TreeNode*>stk;
            TreeNode* ptr=root;
            while(ptr||!stk.empty()){
                if(ptr){
                    stk.push(ptr);
                    ptr=ptr->left;
                }
                else{
                    ptr=stk.top();
                    stk.pop();
                    /* do something */
                    ptr=ptr->right;
                }
            }
        }
        ```

#### iii. 后序遍历

* 细节

        二叉树：左子树->右子树->根
        N叉树：子树1->...->子树n->根

* 代码实现

    * DFS
    
        * 二叉树

            ```c++
            void Postorder(TreeNode* root){
                if(!root)return;
                if(root->left)Inorder(root->left);
                if(root->right)Inorder(root->right);
                /* do something */
            }
            ```

        * N叉树

            ```c++
            void Preorder(Node* root){
                if(!root)return;
                for(auto &child:root->children){
                    Preorder(child);
                }
                /* do something */
            }
            ```

    * BFS

        * 二叉树

            ```c++
            void Postorder(TreeNode* root){
                stack<int>tag;
                stack<TreeNode*>stk;
                TreeNode* ptr=root;
                do{
                    while(ptr){
                        //不断找左节点 并记录为第一次遇到
                        stk.push(ptr);
                        tag.push(0);
                        ptr=ptr->left;
                    }
                    if(!tag.empty()){
                        if(tag.top()){
                            //这时候是第三次遇到这个元素
                            /* use stk.top() to do something */
                            stk.pop();
                            tag.pop();
                        }
                        else{
                            ptr=stk.top();
                            ptr=ptr->right;
                            //将当前栈顶的元素记录为第二次遇到
                            tag.top()++;
                        }
                    }
                }while(ptr||!tag.empty());
            }
            ```

        * N叉树
        
            ```c++
            void Postorder(Node* root) {
                unordered_map<Node*,int>cnt;
                stack<Node*>stk;
                Node* ptr=root;
                while(!stk.empty()||ptr!=nullptr) {
                    while(ptr!=nullptr){
                        stk.push(ptr);
                        if(ptr->children.size()){
                            //记录为第一次遇到
                            cnt[ptr]=0;
                            ptr=ptr->children[0];
                        } else {
                            break;
                        }         
                    }
                    ptr=stk.top();
                    //更新遇见次数
                    int index=cnt[ptr]+1;
                    if(index<ptr->children.size()) {
                        cnt[ptr]=index;
                        ptr=ptr->children[index];
                    }else {
                        /* use ptr do something */
                        stk.pop();
                        cnt.erase(ptr);
                        ptr=nullptr;
                    }
                }
            }
            ```

#### iv. 邻接表表示的树的遍历

利用树的性质，遍历树的各个节点一次。

* 代码实现

    ```C++
    void dfs(int u,int pre){
        for(auto &v:next[u]){
            if(v!=pre){
                dfs(v);
            }
        }
    }
    ```

### B. 树的重建

* 利用先序遍历序列与中序遍历序列实现树的重建

    * 代码实现

        ```C++
        TreeNode* TreeBuild(Array<int>& pre_order, Array<int>& in_order, int pl, int pr, int il, int ir) {
            if (il > ir)return nullptr;
            int root_val = pre_order[pl];
            TreeNode* root = new TreeNode(root_val);
            int index_in = in_order.find(root_val);
            if (index_in == -1) {
                exit(-1);
            }
            int left_size = index_in - il;
            root->left = TreeBuild(pre_order, in_order, pl + 1, pl + left_size, il, index_in - 1);
            root->right = TreeBuild(pre_order, in_order, pl + left_size + 1, pr, index_in + 1, ir);
            return root;
        }
        ```
        ```C++
        TreeBuild(pre,in,0,0,0,0);
        ```

### C. 最小（大）生成树

* 普里姆算法

    * 细节

            每次从当前集合的点里找到最短的、可以拓展点集的边将其点与边并入当前集合，不断贪心地进行。
            很像迪杰斯特拉算法的思想，迪杰斯特拉是以到点最小，而普里姆是集合到点最小，后者不能保证点到点最小，但保证了边权和最小。

    * 代码实现
    
        默认从0顶点出发，返回生成树的邻接矩阵。

        ```c++
        auto getMiniSpanTree=[&](vector<vector<pair<int,int>>>&edges) -> vector<vector<int>> {
            int n=edges.size();
            vector<vector<int>>retG(n);
            vector<pair<int,int>>lowest(n,{0x3f3f3f3f,-1});
            priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>qu;
            qu.push({0,0});
            lowest[0]={0,0};
            int num=0;
            while(!qu.empty()){
                auto [cost,u]=qu.top();
                qu.pop();
                if(lowest[u].first<cost)continue;
                lowest[u].first=0;
                if(u!=lowest[u].second){
                    retG[u].push_back(lowest[u].second);
                    retG[lowest[u].second].push_back(u);
                }
                num++;
                for(auto &[v,w]:edges[u]){
                    if(lowest[v].first>w){
                        qu.push({w,v});
                        lowest[v]={w,u};
                    }
                }
            }
            return num==n?retG:vector<vector<int>>();
        };
        ```

    * 分析

        * 时间复杂度：O(nlogn)
        
        * 空间复杂度：O(n)

* 克鲁斯卡尔算法

    * 细节

        将边从小到大排序，将可以连接两个不同连通分支的边加入边集。

    * 代码实现

        ```C++
        auto getMiniSpanTree=[&](vector<vector<int>>&edges,int n) -> vector<vector<int>> {
            sort(edges.begin(),edges.end(),[&](const vector<int>&e1,const vector<int>&e2){
                return e1[2]<e2[2];
            });
            UnionFind uf(n);
            vector<vector<pair<int>>>ret(n);
            for(auto &e:edges){
                int u=e[0],v=e[1],w=e[2];
                if(!uf.connect(u,v)){
                    uf.findAndUnite(u,v);
                    ret[u].push_back({v,w});
                }
            }
            return ret;
        }
        ```
    
    * 分析

        * 时间复杂度：O(eloge)
        
        * 空间复杂度：O(n)


### D. 共同祖先问题

* 数组模式

    * 代码实现

        * 从序号1开始

            ```C
            int lowestCommonAncestor(int u,int v){
                while(u!=v){
                    if(u>v)swap(u,v);
                    v>>=1;
                }
                return u;
            }
            ```

        * 从序号0开始

            ```C
            int lowestCommonAncestor(int u,int v){
                while(u!=v){
                    if(u>v)swap(u,v);
                    v=(v-1)/2;
                }
                return u;
            }
            ```

    * 分析

        * 时间复杂度：O(logn)
        
        * 空间复杂度：O(1)

* 指针模式

    * 代码实现

        * 普适

            ```C++
            TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
                if(!root||root==p||root==q)return root;
                TreeNode* left=lowestCommonAncestor(root->left,p,q);
                TreeNode* right=lowestCommonAncestor(root->right,p,q);
                if(left&&right)return root;
                else if(left)return left;
                return right;
            }
            ```

        * 二叉搜索树

            ```C++
            TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
                if(p->val<root->val&&q->val<root->val)return lowestCommonAncestor(root->left,p,q);
                else if(p->val>root->val&&q->val>root->val)return lowestCommonAncestor(root->right,p,q);
                return root;
            }
            ```

    * 分析

        * 时间复杂度：O(n)
        
        * 空间复杂度：O(n)

### E. 将树分为和相等的子树

* 细节

    利用dfs，每次走向一个子节点，计算子节点对此节点的奉献：定义为子节点$sum$不足$target$时奉献就是$sum$，当超过时就是不合法状态返回$-1$，当与$target$相等时自成一组则奉献为0。

* 代码实现

    ```C++
    function<int(int,int,int)> dfs=[&](int u,int pre,int target)->int{
        int cur=val[u];
        for(auto &v:next[u]){
            if(v!=pre){
                int sub=dfs(v,u,target);
                if(sub<0)return -1;
                cur+=sub;
            }
        }
        if(cur>target){
            return -1;
        }
        else if(cur<target){
            return cur;
        }
        return 0;
    };
    ```

## 6. 圈（环）

### A. Floyd 判圈算法

* 细节

    对一个存在圈的图，使用慢指针slow每次走一格、快指针fast每次走两格。
    
    如果存在圈，那他们一定会相遇，此时假设fast走了起点到圈入口的距离a，再绕圈走了n圈，最后走到里圈入口b距离的点与slow相遇，此时此刻slow走了a，再走了b。同时假设还需要走c才能重新回到起点。
    
    $$
    2(a+b)=a+b+n(b+c)
    $$

    可知：

    $$
    a=c+(n−1)(b+c)
    $$

    故此时将slow放回起点且slow和fast每次走一格，slow和fast正好可以在圈的入口相遇。

* 代码实现

    ```C++
    ListNode *detectCycle(ListNode *head) {
        ListNode *fptr=head,*sptr=head;
        do{
            if(fptr==nullptr||fptr->next==nullptr)return nullptr;
            fptr=fptr->next->next;
            sptr=sptr->next;
        }while(fptr!=sptr);
        fptr=head;
        while(fptr!=sptr){
            fptr=fptr->next;
            sptr=sptr->next;
        }
        return fptr;
    }
    ```

* 分析

    * 时间复杂度：O(n)

    * 空间复杂度：O(1)

### B. 无向图最小环问题

* 细节

    当存在(u,v)边，我们可以知道u<->v的最小距离为1。若u、v是一个环中的两个结点，显然删除当前边(u,v)仍能找到一个最短路，最小的圈就一定出现在这个最短路和这条边结合。依次删除边->最短路。

* 代码实现

    ```C++
    int findShortestCycle(int n, vector<vector<int>>& edges) {
        auto Dijkstra=[&](vector<set<int>>&next,int k) -> vector<int> {
            int n=next.size();
            vector<int>dicts(n,0x3f3f3f3f);
            priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>qu;
            qu.push({0,k});
            dicts[k]=0;
            while(!qu.empty()){
                auto [d,u]=qu.top();
                qu.pop();
                if(dicts[u]<d)continue;
                for(auto &v:next[u]){
                    int d1=d+1;
                    if(dicts[v]>d1){
                        qu.push({d1,v});
                        dicts[v]=d1;
                    }
                }
            }
            return dicts;
        };
        vector<set<int>>next(n);
        for(auto &e:edges){
            next[e[0]].insert(e[1]);
            next[e[1]].insert(e[0]);
        }
        int res=INT_MAX;
        for(auto &e:edges){
            next[e[0]].erase(e[1]);
            next[e[1]].erase(e[0]);
            vector<int>d=Dijkstra(next,e[0]);
            if(d[e[1]]!=0x3f3f3f3f){
                res=min(res,d[e[1]]+1);
            }
            next[e[0]].insert(e[1]);
            next[e[1]].insert(e[0]);
        }
        return res==INT_MAX?-1:res;
    }
    ```

* 分析

    * 时间复杂度：O($E^2logE$)
    
    * 空间复杂度：O(E)

# **数论**

## 1. 生成质数问题

* 线性筛

    * 细节

        默认每一个数都是质数，而遍历到一个数的时候将它的倍数全部排除，即将他们标记为非质数。而这其中当 $i\%primes[j]==0$ 满足时可以退出，因为当一个数$ i $可以被 $primes[j]$ 整除，那么对于合数 $i⋅primes[j+1] $而言，它一定在后面遍历到 $(i/primes[j])⋅primes[j+1]$ （这个数一定大于$i$）这个数的时候有$ (i/primes[j])⋅primes[j+1]⋅primes[j]==i⋅primes[j+1]$ 被标记，所以之后的标记都是会在之后进行的，这时候退出是安全的。

    * 代码实现

        ```C++
        vector<int> getPrimes(int n){
            vector<int> primes;
            vector<int> isPrime(n+1,1);
            for(int i=2;i<=n;i++){
                if(isPrime[i])primes.push_back(i);
                for (int j=0;j<primes.size()&&i*primes[j]<=n;j++){
                    isPrime[i*primes[j]]=0;
                    if(i%primes[j]==0){
                        break;
                    }
                }
            }
            return primes;
        }
        ```

    * 分析

        * 时间复杂度：O(n)

        * 空间复杂度：O(n)

## 2. 快速幂

* 细节

        每次计算时，将其二分计算，使得时间对数减少。

* 代码实现

    ```C++
    int pow(long long num,long long n,int mod){
        long long ret=1;
        while(n){
            if(n&1)ret=(ret*num)%mod;
            num=(num*num)%mod;
            n>>=1;
        }
        return ret;
    }
    ```
* 分析

    * 时间复杂度：O(logn)

    * 空间复杂度：O(1)

* 应用

    * 乘法逆元

        根据费马小定理，当p为质数时下式成立。 
        $$
            a^{p-1}\bmod p=1
        $$
        之和可以推导出下式。
        $$
            a^{p-2}=a^{-1}\\
            a^{-1}=a^{p-2}
        $$
        故可以用乘法来代替除法。

## 3. 快速乘

* 代码实现

    ```C++
    long long mul(long long x, long long y, long long mod){
        long long t = (long double)x / mod * y;
        long long res = (unsigned long long)x * y - (unsigned long long)t * mod;
        return (res + mod) % mod;
    }
    ```

## 4. 最大公约数

### A. 基本

* 细节

    利用辗转相除法，得到最大公约数。

    $$
    \left.\begin{matrix} 
    a>b \\
    a \div b = q \dots r  \Rightarrow  a=bq+r  \Rightarrow  r=a-bq\\
    gcd(a,b)=d \Rightarrow a=dm,b=dn\\
    \end{matrix}\right\}
    \Rightarrow r=dm-dnq \Rightarrow r=d(m-nq) \Rightarrow d|r\\
    \Rightarrow gcd(a,b)==gcd(b,r)
    $$
    
    b==0 -> 直接返回a
    b!=0 -> 计算b和a%b的最大公约数（当a<b,操作等同于gcd(b,a)保证了计算的是a>b的情况。）


* 实现

    ```C++
    inline int gcd(int a,int b) {
        int r;
        while(b>0){
            r=a%b;
            a=b;
            b=r;
        }
        return a;
    }
    ```

    ```C++
    inline long long gcd(long long a, long long b) {
        return b > 0 ? gcd(b, a % b) : a;
    }
    ```

* 分析

    * 时间复杂度：O(logn)

    * 空间复杂度：O(1)/O(logn)

### B. 运用

#### a. 裴蜀定理

$$
\gcd(x,y)=d \Longrightarrow \exists{a、b},使得 (ax+by)\mod{d}=0 \\
\gcd(x,y)=1 \Longrightarrow \exists{a、b},使得ax+by=1 
$$

对于n维来说同样成立。

$$
\gcd(x_{1},...,x_{n})=1 \Longrightarrow \exists{a_{1},...,a_{n}},使得a_{1}x_{1}+...+anxn=1
$$

* 应用：

    * https://leetcode.cn/problems/check-if-it-is-a-good-array/

## 5. 分解质因数

* 细节

    能除就除，不能就跳过，非质数其因子一定被耗尽。

* 代码实现

    ```C++
    vector<vector<int>> divide(int n){
        vector<vector<int>>ret;
        for(int i=2;i*i<=n;i++){
            if(n%i==0){
                int cnt=0;
                while(n%i==0){
                    n/=i;
                    cnt++;
                }
                ret.push_back({i,cnt});
            }
        }
        if(n!=1)ret.push_back({n,1});
        return ret;
    }
    ```

## 6. 分解因子

* 代码实现

    ```C++
    for(int i=1;i*i<=n;i++){
        if(n%i==0){
            ret.push_back(i);
            ret.push_back(n/i);
        }
    }
    ```

## 7. 二进制处理

可以使用bitset<N>来处理巨量的二进制数。

### A. 快速获取二进制数1的数目

* 细节

    注意到 $ num\&(num-1) $ 将num的最低位1反转，其余不变。

* 代码实现

    ```C++
    int numOf1(int num){
        int res=0;
        while(num!=0){
            res++;
            num=num&(num-1);
        }
        return res;
    }
    ```

    ```C++
    auto numOf1=[&](int num){
        int res=0;
        while(num!=0){
            res++;
            num=num&(num-1);
        }
        return res;
    };
    ```

### B. 快速获取二进制子集

* 细节

    枚举二进制数 $ (1011)_{B} $ 的子集形如 $ (1001)_{B} $ 、$ (1010)_{B} $ 的数字。其思路其实是和A差不多的。比起枚举并判断是否是其子集更快的方法。

* 代码实现

    ```C++
    int bit;
    /* assign bit */
    for(int mask=bit;mask;mask=(mask-1)&bit){
        /* do somethings */
    }
    ```

### C. 仅保留最右边1

* 代码实现

    ```C++
    static constexpr int lowbit(int x) {
        return x & (-x);
    }
    ```

### D. 格雷码编码

* 代码实现

    ```C++
    vector<int> grayCode(int n) {
        vector<int> ret(1 << n);
        for (int i = 0; i < ret.size(); i++) {
            ret[i] = (i >> 1) ^ i;
        }
        return ret;
    }
    ```

    ```C++
    vector<int> getGray(int n, int start) {
        vector<int> ret(1 << n);
        for (int i = 0; i < ret.size(); i++) {
            ret[i] = (i >> 1) ^ i ^ start;
        }
        return ret;
    }
    ```

## 8. 矩阵运算

### A. 矩阵乘法

* 代码实现

    ```C++
    auto matrixMul = [&](vector<vector<int>>& a, vector<vector<int>>& b,int mod)->vector<vector<int>> {
        int m = a.size(), p = a[0].size(), n = b[0].size();
        vector<vector<int>>res(m, vector<int>(n));
        for (int i = 0; i < m; ++i){
            for (int j = 0; j < n; ++j){
                for (int k = 0; k < p; ++k){
                    res[i][j] = ((ll)(a[i][k] % mod) * (b[k][j] % mod) + res[i][j]) % mod;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res[i][j] = (res[i][j] + mod) % mod;
            }
        }
        return res;
    };
    ```

* 分析

    *  时间复杂度：O(mpn)

## 9. $n^n$的性质

* 最右边一位 => 快速幂mod取10

* 最左边一位 => 对数性质

    $n^n=10^{\log_{10}{n^n}}=10^{n\log_{10}{n}}=10^{N+s}=10^N10^s$

    最高一位取决于$10^s$
    
    也即是$n\log_{10}{n}$的小数部分。

## 10. 组合数/阶乘

### 维护方法

* “+”法维护

    利用多数题目只需要利用扁平的组合数范围，可以直接递推维护。

    ```C++
    vector<vector<int>>C(N+1,vector<int>(M+1));
    C[0][0]=C[1][0]=C[1][1]=1;
    for(int i=2;i<=N;i++){
        C[i][0]=1;
        for(int j=1;j<=min(M,i);j++){
            C[i][j]=(C[i-1][j]+C[i-1][j-1])%mod;
        }
    }
    ```

* “x”法维护

    利用多数题目限制对$mod$取模，可以选择维护乘法逆元。

    维护出需要的所有$A$，同时维护出所有对应的$A_f$，在需要的时候进行如下计算：

    ${n \choose m}=\frac{n!}{m!(n-m)!}=\frac{A[n]}{A[m]A[n-m]}=A[n]A_f[m]A_f[n-m]$

    ```C++
    auto fpow=[&](long long num,long long n,int mod){
        long long ret=1;
        while(n){
            if(n&1)ret=(ret*num)%mod;
            num=(num*num)%mod;
            n>>=1;
        }
        return ret;
    };
    vector<int>A(N+1),Af(N+1);
    A[0]=A[1]=Af[0]=Af[1]=1;
    for(long long i=2;i<=N;i++){
        A[i]=(i*A[i-1])%mod;
        Af[i]=fpow(A[i],mod-2,mod);
    }
    auto nCm=[&](int n,int m){
        return ((long long)(((long long)A[n]*A_f[m])%mod)*A_f[n-m])%mod;
    };
    ```

### 使用示例

#### A. 隔板法

将一些物品分配到一些地方的可能情况的数目。

* 每个地方必须最少有一个

         1 2 3 4 5 6 7        
         O O O O O O O
          1 2 3 4 5 6

    如上图，$n$个物品之间有$n-1$个隔板，若要分成$m$组，其意义相当于从$n-1$个隔板选择$m-1$个，这样能保证所有组都有。故答案为${n-1 \choose m-1}$。

* 有的地方可以没有物品

    $n$个物品分为$m$组，有的组可以不放，这个问题可以转化为先给物品加入$m-1$个“空物品”，这样问题就变为了${n+m-1 \choose m-1}={n+m-1 \choose n}$
    

# **字符串**

## 1. 字符串匹配算法KMP

* 细节

        每次比较失效时会仅移动模式串P的指针而移动主串T的指针，减少无效对比。每次失效的时候，查找next数组，寻找模式串下一个应该在的位置。
        
            0 1 2 3 4 5 6 7 8 9              0 1 2 3 4 5 6 7 8 9
        P   = = - - = = X - -           ->           = = ? - = = - - -      
        T   - - - - - - - - - - - - -   ->   - - - - - - - - - - - - -
        
        当模式串在比较下标6位置出现失配时，假定6下标位置前面两个字符与模式串前两个字符一致，那就可以如上图一样移动模式串再次比较。这个next[6]存储的就是当P[6]!=T[6]时，模式串的指针应该移动到哪里？这个问题的答案其实就是不重叠前后缀最大时前缀最后一个元素的后一个元素的位置。这样可以令主串T的指针不用来回走。
        
        实际例子：
        
                0 1 2 3 4 5 6 7 8 9
        P     * A B A A B C A B A D
        
        next   -1 0 0 1 1 2 0 1 2 3 0

    $$
    \begin{aligned}
    next[j] = 
    \left\{ \begin{matrix}&-1,j=0 \\
    &max\{k|p[0:k-1]=p[j-k+1:j]\},k\ne\emptyset \\
    &0,other
    \end{matrix}
    \right.
    \end{aligned}
    $$

        我们假设P[-1]是一个通配符，可与任何字符作配对。P[0]错了就需要将本次比较的这个地方的字符与*作配对，所以next[0]=-1。假设正在计算next[j]，k=next[j-1]，如果P[k]==P[j-1]那就直接是next[j]=k+1，因为P[0:k-1]严格等于P[j-1-k:j-2]；当P[k]!=P[j-1]可以预见的是前缀和后缀要相等就不能那么长了，但我们由于上一次的k还相对大，P[0:k-1]严格等于P[j-1-k:j-2]，k变小仍满足，故k回退到next[k]（为什么对？见下图），再次比较，最终得到可以满足P[k]==P[j-1]的k，完成求解（注意边界k==-1）。
        
                 K       j-1               next[k]
                 |        |                  |
        S -> [S1]X[--][S2]O      S1 -> [S1_1]X[--][S1_2]      S2 -> [S2_1]X[--][S2_2]
        
        一定有S1==S2，故S1_2==S2_2，故找到next[k]作为下一次的比较对象是安全的，[S1_1]与next[k]对应的X以及[S2_2]与j-1对应的O正是这个子序列的前缀和后缀。


* 代码实现

    * next数组(找最大相同前后缀问题)

        ```c++
        vector<int> get_KMP_next(string s){
            //ptr维护j-1 prek维护上一个k
            int n=s.size(),ptr=0,prek=-1;
            vector<int>ret(n+1,-1);
            while(ptr<n){
                //遇到通配符或者是得到配对
                if(prek==-1||s[ptr]==s[prek]){
                    ret[++ptr]=++prek;
                }
                //寻找上一个k
                else prek=ret[prek];
            }
            return ret;
        }
        ```

        ```c++
        auto get_next=[&](string _str)->vector<int> {
            int _n=_str.size(),_ptr=0,_prek=-1;
            vector<int>_ret(_n+1,-1);
            while(_ptr<_n){
                if(_prek==-1||_str[_ptr]==_str[_prek]){
                    _ret[++_ptr]=++_prek;
                }
                else{
                	_prek=_ret[_prek]; 
                }
            }
            return _ret;
        };
        ```

    * 利用next数组完成查找

        ```c++
        int find_KMP(string t,string p){
            int tptr=0,pptr=0,tn=t.size(),pn=p.size();
            vector<int>next=get_KMP_next(p);
            while(tptr<tn&&pptr<pn){
                if(pptr<0||t[tptr]==p[pptr]){
                    tptr++,pptr++;
                }
                else{
                    pptr=next[pptr];
                }
            }
            if(pptr==pn)return tptr-pn;
            else return -1;
        }
        ```
    
    * 封装

        ```c++
        int getIndex(string t,string p){
            int tptr=0,pptr=0,tn=t.size(),pn=p.size();
            auto get_next=[&](string _str)->vector<int> {
                int _n=_str.size(),_ptr=0,_prek=-1;
                vector<int>_ret(_n+1,-1);
                while(_ptr<_n){
                    if(_prek==-1||_str[_ptr]==_str[_prek]){
                        _ret[++_ptr]=++_prek;
                    }
                    else{
                    _prek=_ret[_prek]; 
                    }
                }
                return _ret;
            };
            vector<int>next=get_next(p);
            while(tptr<tn&&pptr<pn){
                if(pptr<0||t[tptr]==p[pptr]){
                    tptr++,pptr++;
                }
                else{
                    pptr=next[pptr];
                }
            }
            if(pptr==pn)return tptr-pn;
            else return -1;
        }
        ```

* 分析

    * 时间复杂度： O(n+m)

    * 空间复杂度： O(m)

## 2. 回文串

### A. 快速判断某区间是否回文串

* 细节

    直接使用二维动态规划，$dp[i][j]=dp[i+1][j-1]\&\&s[i]==s[j]$来解决。

### B. 只需要最长的回文串

* 细节

    * 加入'#'来确保每一个回文串的长度都是奇数，不需要奇数偶数分来讨论。

    * 如果当前位置是之前半径范围内 -> 查看上一次半径中心为中心点的对称位置 -> 找到不需要对比的最大边界

            ===O==X==O===
            
               |      | |
            
             i_sym    i r
    
    * 如果不在之前半径范围内 -> 对当前位置进行扩张

* 代码实现

    ```C++
    int expand(const string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return (right - left - 2) / 2;
    }

    string longestPalindrome(string s) {
        int start = 0, end = -1;
        string t = "#";
        for (char c: s) {
            t += c;
            t += '#';
        }
        t += '#';
        s = t;

        vector<int> arm_len;
        int right = -1, j = -1;
        for (int i = 0; i < s.size(); ++i) {
            int cur_arm_len;
            if (right >= i) {
                int i_sym = j * 2 - i;
                int min_arm_len = min(arm_len[i_sym], right - i);
                cur_arm_len = expand(s, i - min_arm_len, i + min_arm_len);
            } else {
                cur_arm_len = expand(s, i, i);
            }
            arm_len.push_back(cur_arm_len);
            if (i + cur_arm_len > right) {
                j = i;
                right = i + cur_arm_len;
            }
            if (cur_arm_len * 2 + 1 > end - start) {
                start = i - cur_arm_len;
                end = i + cur_arm_len;
            }
        }

        string ans;
        for (int i = start; i <= end; ++i) {
            if (s[i] != '#') {
                ans += s[i];
            }
        }
        return ans;
    }
    ```

* 分析

    * 时间复杂度：O(n)

    * 空间复杂度：O(n)

## 3. 最大回文子串

* 代码实现

    ```C++
    for(int i=n-1;i>=0;i--){
        dp[i][i]=1;
        for(int j=i+1;j<n;j++){
            if(s[i]==s[j]){
                dp[i][j]=dp[i+1][j-1]+2;
            }
            else{
                dp[i][j]=max(dp[i][j-1],dp[i+1][j]);
            }
        }
    }
    ```

# **二分查找/折半查找**

## 1. 查找一个内容的上界

upper_bound，第一个大于查找元素的位置。

* 代码实现

    ```C++
    int upper_bound(int LOW,int HIGH,int target,vector<int>&arr){
        int l=LOW,r=HIGH;
        while(l<=r){
            int mid=(r-l)/2+l;
            if(arr[mid]<=target){
                l=mid+1;
            }
            else{
                r=mid-1;
            }
        }
        return l;
    }
    ```

## 2. 查找一个内容的下界

lower_bound，第一个大于等于的查找元素的位置。

* 代码实现

    ```C++
    int lower_bound(int LOW,int HIGH,int target,vector<int>&arr){
        int l=LOW,r=HIGH;
        while(l<=r){
            int mid=(r-l)/2+l;
            if(arr[mid]<target){
                l=mid+1;
            }
            else{
                r=mid-1;
            }
        }
        return l;
    }
    ```

## 3. 查找单调函数的零点

* 代码实现

    ```C++
    auto check=[&](int num){
        if(/* */)return true;
        else return false;
    };
    int l=LOW,r=HIGH;
    while(l<r){
        int mid=(r-l)/2+l;
        if(check(mid))r=mid;
        else l=mid+1;
    }
    ```

    这里mid偏l所以不会有无限循环风险。当mid成立的时候可能是答案，所以mid需要保留，当mid不成立的时候mid不必保留。

## 4. 查找凹凸函数的极值点

以凸函数为例。

* 代码实现

    ```C++
    auto calculate=[&](int num){
        /* */
        return val;
    };
    int l=LOW,r=HIGH;
    while(l<r){
        int mid=(l+r)>>1;
        if(calculate(mid)>calculate(mid+1))r=mid;
        else l=mid+1;
    }
    ```

    不断靠近最大值。

# **排序算法**

## 1. 插入排序

插入排序一次排序中不一定能使一个元素到达它最终位置！

### A. 直接插入排序

* 细节

        每次将下一个元素插入前面已经有序的序列，每一次都能获得记录数+1的有序表。一个元素已经有序，所以不必再进行插入，可以直接从第二个元素开始。0号元素可以作为哨兵使用，每次将下一个元素插入前可以将这个元素先拷贝一份到哨兵位置，这样一定有匹配的地方。

* 代码实现

    ```c++
    void InsertSort(vector<int>&arr){
        //arr从1号位置开始存储信息
        //arr.size()这里应该按照题意来，可以替换成<=元素个数n
        for(int i=2;i<arr.size();i++){
            //arr[i-1]<=arr[i]时已经有序
            if(arr[i-1]>arr[i]){
                //设置哨兵
                arr[0]=arr[i];
                //直接j=i-1会导致比较次数多一次
                arr[i]=arr[i-1];
                int j=i-2;
                //只要前面的比自己大就将其后移
                while(arr[j]>arr[0]){
                    //边查找变移动
                    arr[j+1]=arr[j];
                    j--;
                }
                //多移动了一位才退出循环所以是j+1
                arr[j+1]=arr[0];
            }
        }
    }
    ```

* 分析

    * 稳定排序，每次比较将能插入就马上插入。

    * 时间复杂度：最好O(n) 最坏O($n^2$)

            最好的情况是数组已经有序，只需要比较n-1次就完成了排序
            最坏的情况是数组逆序排列，每次插入元素都要找到哨兵才能停止。需要比较∑(2,n)i次也就是(n+2)(n-1)/2次，移动∑(2,n)i+1也就是(n+4)(n-1)/2次。
            引入二分也难以避免移动带来的开销。

    * 空间复杂度： O(1)

### B. 希尔排序

* 细节

        将数组分为若干个数组，并不断减少分组数，最后分组数为1时数组已经基本有序。希尔排序的分组序列是关键。

* 代码实现

    ```c++
    void ShellInesert(vector<int>&arr,int dk){
        //arr从1号位置开始存储信息
        //0号位置用于暂存
        //dk是当前的分组个数
        //arr.size()这里应该按照题意来，可以替换成<=元素个数n
        //这里只不过是把增量变为不确定的值
        //i=dk+1指向的是一个分组中的第二个元素，对各组间进行异步的插入排序
        for(int i=dk+1;i<arr.size();i++){
            if(arr[i]<arr[i-dk]){
                arr[0]=arr[i];
                arr[i]=arr[i-dk];
                int j=i-2*dk;
                while(j>0&&arr[0]<arr[j]){
                    arr[j+dk]=arr[j];
                    j-=dk;
                }
                arr[j+dk]=arr[0];
            }
        }
    }
    
    void ShellSort(vector<int>&arr,vector<int>&dlta){
        //dlta是增量的递减序列
        for(auto &dk:dlta){
            ShellInesert(arr,dk);
        }
    }
    ```

* 封装

    ```c++
    auto ShellSort = [&](vector<int>& arr)->void {
        auto ShellInesert = [&](vector<int>& arr, int dk)->void {
            //dk是当前的分组个数
            //i=dk指向的是一个分组中的第二个元素，对各组间进行异步的插入排序
            int temp;
            for (int i = dk; i < arr.size(); i++) {
                if (arr[i] < arr[i - dk]) {
                    temp = arr[i];
                    arr[i] = arr[i - dk];
                    int j = i - 2 * dk;
                    while (j >= 0 && temp < arr[j]) {
                        arr[j + dk] = arr[j];
                        j -= dk;
                    }
                    arr[j + dk] = temp;
                }
            }
        };
        //dlta是增量的递减序列，需要修改
        vector<int>dlta{ 15,7,3,1 };
        for (auto& dk : dlta) {
            ShellInesert(arr, dk);
        }
    };
    ```

* 分析

    * 不稳定排序，可能将相等的元素位置颠倒

    * 时间复杂度主要取决于dlta的选取，研究表明当$dlta[k]=2^{dn-k+1}-1$时时间复杂度可以为O($n^{3/2}$)，也要注意dlta各元素要互质。

    * 空间复杂度： O(1)


## 2. 冒泡排序

* 细节

        每次将相邻元素相互比较，将特定元素往后移。

* 代码实现

    ```c++
    void BubbleSort(vector<int>&arr){
        bool flag;
        for(int i=arr.size();i>0;i--){
            flag=0;
            for(int j=1;j<i;j++){
                if(arr[j-1]>arr[j]){
                    swap(arr[j],arr[j-1]);
                    flag=1;
                }
            }
            if(!flag)break;
        }
    }
    ```

* 分析

    * 稳定排序

    * 时间复杂度： 最好O(n) 最差O($n^2$)

            最好时是序列已经有序，只需要比较n-1次。
            最坏时是序列逆序排列，需要比较∑(2,n)(i-1)也就是n(n-1)/2次。

    * 空间复杂度： O(1)


## 3. 快速排序

快速排序趟排序至少能找到一个正确的位置（枢轴）。

快速排序的nlogn常数很小。

* 细节
        
  
      快速排序主要是利用分治的思想，将数组分为一个个数组，对每个数组选定一个枢轴，将其中元素分在枢轴两端，然后将枢轴两端元素分别再分割为两个数组再如此操作。
  
* 代码实现

    * 固定枢轴

        * 代码实现

            ```c++
            int Partition(vector<int>&arr,int low,int high){
                //是从1号位置开始存储信息
                //0号位置存储枢轴信息
                //固定low位置作为枢轴
                arr[0]=arr[low];
                int pivotkey=arr[0];
                while(low<high){
                    //可以保证的是每次循环结束时low总是指向枢轴
                    //从右端找到第一个小于枢轴的位置与枢轴位置交换。
                    while(low<high&&arr[high]>=pivotkey)high--;
                    //枢轴位置不用赋值，不一定是最后位置
                    arr[low]=arr[high];
                    while(low<high&&arr[low]<=pivotkey)low++;
                    arr[high]=arr[low];
                }
                arr[low]=pivotkey;
                return low;
            }
            
            void Selected(vector<int>&arr,int low,int high){
                if(low<high){
                    int pivotloc=Partition(arr,low,high);
                    Selected(arr,low,pivotloc-1);
                    Selected(arr,pivotloc+1,high);
                }
            }
            
            void qSort(vector<int>&arr){
                Selected(arr,1,arr.size()-1);
            }
            ```

        * 不稳定排序。

        * 时间复杂度： 最好O(nlogn) 最坏O($n^2$)
    
                最好出现在数组基本无序。
                最坏出现在数组基本有序，这样枢轴每次都不能移动。

        * 空间复杂度： 由于每个枢轴都要一次调用 最好O(logn) 最坏O(n)

    * 引入随机枢轴

        * 代码实现
    
            ```c++
            int partition(vector<int>& nums,int l,int r) {
                //0号位置开始存储信息
                int pivot=nums[r];
                //i维护的是最后一个小于等于枢轴的位置 i+1就是枢轴的最后位置
                int i=l-1;
                //r已经是当前枢轴的位置，不必比较了
                for (int j=l; j<=r-1;j++) {
                    //当遍历时发现小于等于枢轴的元素将其与i+1交换位置
                    //保证i+1填入枢轴是正确的
                    if (nums[j]<=pivot) {
                        i=i+1;
                        swap(nums[i],nums[j]);
                    }
                }
                //将枢轴放在正确位置
                swap(nums[i+1],nums[r]);
                return i+1;
            }
            
            int randomized_partition(vector<int>& nums,int l,int r) {
                //随机选择一个下标作为枢轴
                int i=rand()%(r-l+1)+l;
                //把选出来的枢轴放到最右边
                swap(nums[r],nums[i]);
                return partition(nums, l, r);
            }
            
            void randomized_selected(vector<int>& arr,int l,int r) {
                if (l<r) {
                    int pos = randomized_partition(arr, l, r);
                    randomized_selected(arr, l, pos - 1);
                    randomized_selected(arr, pos + 1, r);
                }
                
            }
            
            void qSort(vector<int>& arr){
                srand((unsigned)time(NULL));
                randomized_selected(arr,0,arr.size()-1);
            }
            ```
    
        * 封装
        
            ```c++
            auto Qsort = [&](vector<int>& arr)->void {
                auto partition = [&](vector<int>& nums, int l, int r)->int {
                    int pivot = nums[r];
                    //i维护的是最后一个小于等于枢轴的位置 i+1就是枢轴的最后位置
                    int i = l - 1;
                    //r已经是当前枢轴的位置，不必比较了
                    for (int j = l; j <= r - 1; j++) {
                        //当遍历时发现小于等于枢轴的元素将其与i+1交换位置
                        //保证i+1填入枢轴是正确的
                        if (nums[j] <= pivot) {
                            i = i + 1;
                            swap(nums[i], nums[j]);
                        }
                    }
                    //将枢轴放在正确位置
                    swap(nums[i + 1], nums[r]);
                    return i + 1;
                };
            
                auto randomized_partition = [&](vector<int>& nums, int l, int r)->int {
                    //随机选择一个下标作为枢轴
                    int i = rand() % (r - l + 1) + l;
                    //把选出来的枢轴放到最右边
                    swap(nums[r], nums[i]);
                    return partition(nums, l, r);
                };
            
                function<void(vector<int>&, int, int)> randomized_selected = [&](vector<int>& arr, int l, int r) {
                    if (l < r) {
                        int pos = randomized_partition(arr, l, r);
                        randomized_selected(arr, l, pos - 1);
                        randomized_selected(arr, pos + 1, r);
                    }
                };
            
                srand((unsigned)time(NULL));
                randomized_selected(arr, 0, arr.size() - 1);
            };
            ```
        
        * 分析
        
            基本与前面一致，但更多时候时间复杂度为O(nlogn)，空间复杂度为O(logn)。

## 4. 选择排序

### A. 简单选择排序

* 细节

        从无序序列中找到最值加入有序序列。

* 代码实现

    ```c++
    void SelectSort(vector<int>&arr){
        for(int i=0;i<arr.size();i++){
            int j=min_element(arr.begin()+i,arr.end())-arr.begin();
            if(i!=j){
                swap(arr[i],arr[j]);
            }
        }
    }
    ```

* 分析

    * 一般来说是不稳定的，也可以做成稳定。

    * 时间复杂度： O($n^2$)
            
            最多移动元素出现在每两个元素都要交换，每次交换需要一个temp，导致每次产生3次交换次数，共计3(n-1)次；最少出现在已经有序，需要移动0次。
            比较次数为∑(1,n)(n-i)也就是n(n-1)/2。

    * 空间复杂度： O(1)

### B. 堆排序

* 细节

        对于所有子树元素都满足大于或小于根节点的树称为堆，堆顶可作为本次输出元素，维护下一个输出元素时可以利用之前取得的信息。

* 代码实现

    ```C++
    void HeapAdjust(vector<int>&arr,int cur,int m){
        //本次操作之前[cur:m]除去cur节点其余已经具备堆的特性
        //本次操作是将[cur:m]完全变为堆
        //这里的写法是针对元素由1号开始存储来编写，左子树2*i，右子树2*i+1
        //若从0号位置开始存储，左子树为2*i+1,右子树为2*i+2
        //cur维护的是新加入堆的元素位置
        int rootval=arr[cur];
        for(int i=2*cur;i<=m;i*=2){
            //选择最大的一颗子树节点
            if(i+1<=m&&arr[i]<arr[i+1])i++;
            //若大于(小于)当前根节点就将其上移,否则已经找到最终位置
            if(rootval>=arr[i])break;
            arr[cur]=arr[i];
            cur=i;
        }
        arr[cur]=rootval;
    }
    
    void HeapSort(vector<int>&arr){
        int n=arr.size()-1;
        //n/2下标是最后一个含有叶子的子树，自底而上建立堆
        //自顶而下很难保证每一次都能在正确的位置
        for(int i=n/2;i>0;i--){
            HeapAdjust(arr,i,n);
        }
        //i==1的时候没必要再来一次了
        //每次把堆顶元素放到最后再将一个叶子放在堆顶再重建堆完成排序
        for(int i=n;i>1;i--){
            swap(arr[i],arr[1]);
            HeapAdjust(arr,1,i-1);
        }
    }
    ```

* 封装

    ```C++
    auto HeapSort = [&](vector<int>& arr,int k)->void {
        auto cmp = [&](const int& a, const int& b)->bool {
            if (a >= b)return 1;
            else return 0;
        };

        auto HeapAdjust = [&](vector<int>& arr, int cur, int m) {
            int rootval = arr[cur];
            //注意i=2*i+1
            for (int i = 2 * cur + 1; i <= m; i = 2 * i + 1) {
                if (i + 1 <= m && cmp(arr[i], arr[i + 1]))i++;
                if (!cmp(rootval,arr[i]))break;
                arr[cur] = arr[i];
                cur = i;
            }
            arr[cur] = rootval;
        };

        int n = arr.size();
        for (int i = (n-2) / 2; i >= 0; i--) {
            HeapAdjust(arr, i, n - 1);
        }
        for (int i = n - 1; i >= 1; i--) {
            swap(arr[i], arr[0]);
            HeapAdjust(arr, 0, i-1);
        }
    };
    ```

* 分析

    * 不稳定排序。

    * 时间复杂度： O(nlogn)

    * 空间复杂度： O(1)

## 5. 归并排序

* 细节

        将两个有序数组重新组合只需要耗费线性，利用分治的思想，将一个数组不断切分，找到切分为由一个元素组成的序列，再和其余序列合并，不断合并最终有序。

* 代码实现

    ```C++
    void Merge(vector<int>partOrdered,vector<int>&arr,int f,int m,int e){
        //这个地方partOrdered一定要开新空间，没法避免
        //将partOrdered[f:m]和partOrdered[m+1:e]并入为有序的arr[f:e]
        //p维护的是加入arr中的位置，s维护的是第二部分需要并入的位置
        //p不用比较了，一定不会越界
        int p=f,s=m+1;
        while(f<=m&&s<=e){
            if(partOrdered[f]<=partOrdered[s]){
                arr[p++]=partOrdered[f++];
            }
            else arr[p++]=partOrdered[s++];
        }
        while(f<=m){
            arr[p++]=partOrdered[f++];
        }
        while(s<=m){
            arr[p++]=partOrdered[s++];
        }
    }
    
    void Msort(vector<int>&pre,vector<int>&ordered,int l,int r){
        //这个地方pre可以不开空间，当ordered被修改时，pre的对应位置也一定不再使用了
        //将pre[l:r]归并排序后放入ordered
        if(l==r){
            //递归退出条件
            ordered[l]=pre[l];
        }
        else{
            //一分为二
            int mid=(r-l)/2+l;
            //将左半的结果放进ordered
            Msort(pre,ordered,l,mid);
            //将右半的结果放进ordered
            Msort(pre,ordered,mid+1,r);
            //合并两个结果
            Merge(ordered,ordered,l,mid,r);
        }
    }
    
    void MergeSort(vector<int>&arr){
        Msort(arr,arr,0,arr.size()-1);
    }
    ```

* 分析

    * 稳定排序。

    * 时间复杂度： O(nlogn)

    * 空间复杂度： O(nlogn)
            
      
            申请了logn次（由于二分）n大小的辅助空间。

## 6. 基数排序

* 细节

        利用了LSD的原理，将每个关键字位的比较叠加，取消了对关键字与关键字之间的比较，先分配再收集完成排序。LSD的排序要求每次排序都是稳定的。利用f和e数组分别维护关于同关键字位的序列中头与尾。

* 代码实现

    ```C++
    #define NUM_D 3
    #define NUM_R 10
    
    void Distribute(vector<vector<int>>&arr,int time,vector<int>&f,vector<int>&e){
        //arr这里是一个静态链表，其中零号位置作为头指针
        //这里没有初始化e，因为e在使用到的时候一定会刷新
        for(int i=0;i<NUM_R;i++){
            f[i]=0;
        }
        //p==0时指向头指针时退出
        for(int p=arr[0][1];p;p=arr[p][1]){
            int j=(arr[p][0]/time)%NUM_R;
            if(!f[j])f[j]=p;
            else{
                arr[e[j]][1]=p;
            }
            e[j]=p;
        }
    }
    
    void Collect(vector<vector<int>>&arr,vector<int>&f,vector<int>&e){
        //ptr维护的是当前的收集到的位数
        //pre维护的是上一个非空子表的末尾元素
        //pre初始化为0最省事
        int ptr=0,pre=0;
        while(ptr<10){
            while(ptr<10&&!f[ptr])ptr++;
            if(ptr<10){
                //将两子表头尾相接
                arr[pre][1]=f[ptr];
                pre=e[ptr];
            }
            ptr++;
        }
        arr[pre][1]=0;
    }
    
    void RadixSort(vector<int>&arr){
        int n=arr.size();
        //新建静态链表
        vector<vector<int>>ptr_arr(n+1,vector<int>(2));
        for(int i=0;i<n;i++){
            ptr_arr[i+1][0]=arr[i];
            ptr_arr[i][1]=i+1;
        }
        vector<int>f(NUM_R),e(NUM_R);
        int time=1;
        for(int i=0;i<NUM_D;i++){
            Distribute(ptr_arr,time,f,e);
            Collect(ptr_arr,f,e);
            time*=NUM_R;
        }
        int p=ptr_arr[0][1];
        for(int i=0;i<n;i++){
            arr[i]=ptr_arr[p][0];
            p=ptr_arr[p][1];
        }
    }
    ```

* 分析

    * 稳定排序。但注意很难实现对double的排序。

    * 时间复杂度： O(d(n+r))
            
      
          需要执行d次收集和分配，其中分配需要n，分配需要r（每一次需要看完r个f）。
      
    * 空间复杂度： 一般为O(r)，但本处为O(n+r)。
    
            新建两个大小为r的数组f、e；若本来就是静态链表n也可以避免。

# **高级数据结构**

## 1. 并查集

* 细节
  
    面对配对、分组问题，快速分组整合。

        vector<int> parent;     记录父节点
        vector<int> size;       记录当前子树的大小(优化树)
        int n;                  顶点数
        int setCount;           集合数
          
        UnionFind(int)          初始化，将每个元素的父节点定义为自己，树大小定义为1
        findset(int)            返回对应顶点编号的父节点
        unite(int,int)          将两个集合合并(两个集合的父节点合并)
        findAndUnite(int,int)   将两个集合合并如已经在同一个集合内返回false，否则返回true。

* 封装

    ```c++
    class UnionFind {
    public:
        vector<int> parent;
        vector<int> size;
        int n;
        int setCount;
    
    private:
        void unite(int x, int y) {
            if (size[x] < size[y]) {
                swap(x, y);
            }
            parent[y] = x;
            size[x] += size[y];
            --setCount;
        }

    public:
        UnionFind(int _n): n(_n), setCount(_n), parent(_n), size(_n, 1) {
            iota(parent.begin(), parent.end(), 0);
        }
        
        int findset(int x) {
            return parent[x] == x ? x : parent[x] = findset(parent[x]);
        }
        
        bool findAndUnite(int x, int y) {
            int parentX = findset(x);
            int parentY = findset(y);
            if (parentX != parentY) {
                unite(parentX, parentY);
                return true;
            }
            return false;
        }

        bool connected(int x, int y){
            return findset(x)==findset(y);
        }
    };
    ```

* 分析

    * 时间复杂度：单次合并O(logn) 

    * 空间复杂度：O(n)

## 2. 线段树

区间查询

* 细节

        类二叉搜索树的特性，但每个节点存放一个区间(lo,hi)，计算得mid=(lo+hi)>>1，其左子树存放区间(lo,mid)，其右子树存放区间(mid+1,hi)。而每个区间可以分别计数并由数的性质将其父节点也同时维护。
        
        当[left,right]数据比较离散时，最好映射到(0,x)这个区间，以节省空间，要保证大的数其映射的值也对应大。（排序再分配）
        
        SegNode* build(int left, int right)             初始化一个区间为[left,right]的线段树。
        void insert(SegNode* root, int val)             向树插入val，维护对应区间的值
        int count(SegNode* root, int left, int right)   查找[left,right]区间的总和值

* 代码实现

    * 维护前缀和

        * 单点更新（插入新值）区间查询。

            ```C++
            struct SegNode {
                long long lo, hi;
                int add;
                SegNode* lchild, *rchild;
                SegNode(long long left, long long right): lo(left), hi(right), add(0), lchild(nullptr), rchild(nullptr) {}
            };

            SegNode* SegSuper;

            SegNode* build(int left, int right) {
                SegNode* node = new SegNode(left, right);
                if (left == right) {
                    return node;
                }
                int mid = (left + right) / 2;
                node->lchild = build(left, mid);
                node->rchild = build(mid + 1, right);
                return node;
            }

            void insert(SegNode* root, int val) {
                root->add++;
                if (root->lo == root->hi) {
                    return;
                }
                int mid = (root->lo + root->hi) / 2;
                if (val <= mid) {
                    insert(root->lchild, val);
                }
                else {
                    insert(root->rchild, val);
                }
            }

            int count(SegNode* root, int left, int right) const {
                if (left > root->hi || right < root->lo) {
                    return 0;
                }
                if (left <= root->lo && root->hi <= right) {
                    return root->add;
                }
                return count(root->lchild, left, right) + count(root->rchild, left, right);
            }
            ```

        * 单点更新 区间查询 窗口内第k大元素

            数据范围确定（线段树大小）、点定为1

            ```C++
            class Segment{
                void update(int x,int d,int sl,int sr,int ptr){
                    if(sl==sr){
                        Tree[ptr]+=d;
                        return;
                    }
                    int mid=(sl+sr)/2;
                    if(x<=mid){
                        update(x,d,sl,mid,ptr<<1|1);
                    }
                    else{
                        update(x,d,mid+1,sr,(ptr<<1)+2);
                    }
                    Tree[ptr]=Tree[ptr<<1|1]+Tree[(ptr<<1)+2];
                }
                int quary(int l,int r,int sl,int sr,int ptr){
                    if(l<=sl&&sr<=r){
                        return Tree[ptr];
                    }
                    int mid=(sl+sr)/2;
                    int ans=0;
                    if(l<=mid){
                        ans+=quary(l,r,sl,mid,ptr<<1|1);
                    }
                    if(mid+1<=r){
                        ans+=quary(l,r,mid+1,sr,(ptr<<1)+2);
                    }
                    return ans;
                }
                int find_by_rank(int k,int sl,int sr,int ptr){
                    if(sl==sr){
                        return sl; 
                    }
                    int mid=(sl+sr)/2;
                    if(Tree[ptr<<1|1]>=k){
                        return find_by_rank(k,sl,mid,ptr<<1|1);
                    }
                    return find_by_rank(k-Tree[ptr<<1|1],mid+1,sr,(ptr<<1)+2);
                }
            public:
                int n;
                vector<int>Tree;
                Segment(int n):n(n){
                    Tree.resize(n<<2);
                }
                void update(int x,int d){
                    update(x,d,0,n-1,0);
                }
                int quary(int l,int r){
                    return quary(l,r,0,n-1,0);
                }
                int find_by_rank(int k){
                    return find_by_rank(k,0,n-1,0);
                }
            };
            ```

        * 区间查询 GCD

            ```C++
            class Segment{
            void build(int sl,int sr,int ptr){
                if(sl==sr){
                    Tree[ptr]=nums[sl];
                    return;
                }
                int mid=(sl+sr)/2;
                build(sl,mid,ptr<<1|1);
                build(mid+1,sr,(ptr<<1)+2);
                Tree[ptr]=gcd(Tree[ptr<<1|1],Tree[(ptr<<1)+2]);
            }
            int quary(int l,int r,int sl,int sr,int ptr){
                if(l<=sl&&sr<=r){
                    return Tree[ptr];
                }
                int mid=(sl+sr)/2;
                int ans=-1;
                if(l<=mid){
                    ans=quary(l,r,sl,mid,ptr<<1|1);
                }
                if(mid+1<=r){
                    if(ans==-1)ans=quary(l,r,mid+1,sr,(ptr<<1)+2);
                    else{
                        ans=gcd(ans,quary(l,r,mid+1,sr,(ptr<<1)+2));
                    }
                }
                return ans;
            }
            public:
                int n;
                vector<int>Tree,nums;
                Segment(vector<int> &nums):nums(nums){
                    n=nums.size();
                    Tree.resize(n<<2);
                    build(0,n-1,0);
                }
                int quary(int l,int r){
                    return quary(l,r,0,n-1,0);
                }
            };
            ```

        * 区域维护 区域查询

            ```C++
            template<class T>
            class Segment {
                void build(vector<int>& nums, int l, int r, int ptr) {
                    if (l == r) {
                        Tree[ptr] = nums[l];
                        return;
                    }
                    int mid = (l + r) / 2;
                    build(nums, l, mid, ptr << 1 | 1);
                    build(nums, mid + 1, r, (ptr << 1) + 2);
                    push_up(ptr);
                }

                void push_down(int ptr,T len) {
                    if(Add[ptr] != 0) {
                        int left = ptr << 1 | 1, right = (ptr << 1) + 2;
                        Add[left] += Add[ptr];
                        Add[right] += Add[ptr];
                        Tree[left] += Add[ptr]*(len - len/2);
                        Tree[right] += Add[ptr]*(len/2);
                        Add[ptr] = 0;
                    }
                }

                void push_up(int ptr) {
                    Tree[ptr] = Tree[ptr << 1 | 1] + Tree[(ptr << 1) + 2];
                }
                
                void update(int sl, int sr, int l, int r, int val, int ptr) {
                    T len = r - l + 1;
                    if(sl <= l && r <= sr) {
                        Add[ptr] += val;
                        Tree[ptr] += len*val;
                        return;
                    }
                    push_down(ptr, len);
                    int mid = (l + r) / 2;
                    if(sl <= mid) {
                        update(sl, sr, l, mid, val, ptr << 1 | 1);
                    }
                    if(mid + 1 <= sr) {
                        update(sl, sr, mid + 1, r, val, (ptr << 1) + 2);
                    }
                    push_up(ptr);
                }

                T query(int sl, int sr, int l, int r, int ptr) {
                    if(sl <= l && r <= sr){
                        return Tree[ptr];
                    }
                    push_down(ptr, r - l + 1);
                    int mid = (l + r) / 2;
                    T ans = 0;
                    if(sl <= mid) {
                        ans += query(sl, sr, l, mid, ptr << 1 | 1);
                    }
                    if(mid + 1 <= sr) {
                        ans += query(sl, sr, mid + 1, r, (ptr << 1) + 2);
                    }
                    return ans;
                }

            public:
                int n;
                vector<T>Tree;
                vector<T>Add;

                Segment(vector<int>& nums) {
                    this->n = nums.size();
                    Tree.resize(4 * n);
                    Add.resize(4 * n);
                    build(nums, 0, n - 1, 0);
                }

                void update(int l, int r, int val) {
                    update(l, r, 0, n-1, val, 0);
                }

                T query(int l, int r) {
                    return query(l, r, 0, n-1, 0);
                }
            };
            ```

        * 区间反转 区间查询 （01线段树）

            ```C++
            template<class T>
            class Segment {
                void build(vector<int>& nums, int l, int r, int ptr) {
                    if (l == r) {
                        Tree[ptr] = nums[l];
                        return;
                    }
                    int mid = (l + r) / 2;
                    build(nums, l, mid, ptr << 1 | 1);
                    build(nums, mid + 1, r, (ptr << 1) + 2);
                    push_up(ptr);
                }
            
                void push_down(int ptr,T len) {
                    if(Add[ptr]) {
                        int left = ptr << 1 | 1, right = (ptr << 1) + 2;
                        Add[left] = !Add[left];
                        Add[right] = !Add[right];
                        Tree[left] = len-len/2-Tree[left];
                        Tree[right] = len/2-Tree[right];
                        Add[ptr] = false;
                    }
                }
            
                void push_up(int ptr) {
                    Tree[ptr] = Tree[ptr << 1 | 1] + Tree[(ptr << 1) + 2];
                }
                
                void flip(int sl, int sr, int l, int r, int ptr) {
                    T len = r - l + 1;
                    if(sl <= l && r <= sr) {
                        Add[ptr] = !Add[ptr];
                        Tree[ptr] = len - Tree[ptr];
                    }
                    else {
                        push_down(ptr, len);
                        int mid = (l + r) / 2;
                        if(sl <= mid) {
                            flip(sl, sr, l, mid, ptr << 1 | 1);
                        }
                        if(mid + 1 <= sr) {
                            flip(sl, sr, mid + 1, r, (ptr << 1) + 2);
                        }
                        push_up(ptr);
                    }
                }
            
                T query(int sl, int sr, int l, int r, int ptr) {
                    if(sl <= l && r <= sr){
                        return Tree[ptr];
                    }
                    push_down(ptr, r - l + 1);
                    int mid = (l + r) / 2;
                    T ans = 0;
                    if(sl <= mid) {
                        ans += query(sl, sr, l, mid, ptr << 1 | 1);
                    }
                    if(mid + 1 <= sr) {
                        ans += query(sl, sr, mid + 1, r, (ptr << 1) + 2);
                    }
                    return ans;
                }
            
            public:
                int n;
                vector<T>Tree;
                vector<bool>Add;
            
                Segment(vector<int>& nums) {
                    this->n = nums.size();
                    Tree.resize(4 * n);
                    Add.resize(4 * n);
                    build(nums, 0, n - 1, 0);
                }
            
                void flip(int l, int r) {
                    flip(l, r, 0, n-1, 0);
                }
            
                T query(int l, int r) {
                    return query(l, r, 0, n-1, 0);
                }
            };
            ```

    * 维护区间最大值

        单点更新（修改原有值），区间查询。

        * 数组实现
        
            C++ new操作较慢，数组会相对快

            ```C++
            class Segment {
                int q(int x, int y, int l, int r, int ptr) {
                    if (x <= l && r <= y) {
                        return Tree[ptr];
                    }
                    else if (x > r || y < l) {
                        return INT_MIN;
                    }
                    int mid = (l + r) / 2;
                    return max(q(x, y, l, mid, ptr << 1 | 1), q(x, y, mid + 1, r, (ptr << 1) + 2));
                }

                void build(vector<int>& nums, int l, int r, int ptr) {
                    if (l == r) {
                        Tree[ptr] = nums[l];
                        return;
                    }
                    int mid = (l + r) / 2;
                    build(nums, l, mid, ptr << 1 | 1);
                    build(nums, mid + 1, r, (ptr << 1) + 2);
                    Tree[ptr] = max(Tree[ptr << 1 | 1], Tree[(ptr << 1) + 2]);
                }

                void _update(int place, int val, int l, int r, int ptr) {
                    int mid = (l + r) / 2;
                    if (l == r) {
                        Tree[ptr] = val;
                        return;
                    }
                    if (place <= mid) {
                        _update(place, val, l, mid, ptr << 1 | 1);
                    }
                    else {
                        _update(place, val, mid + 1, r, (ptr << 1) + 2);
                    }
                    Tree[ptr] = max(Tree[ptr << 1 | 1], Tree[(ptr << 1) + 2]);
                }

            public:
                int n;
                vector<int>Tree;

                Segment(vector<int>& nums) {
                    this->n = nums.size();
                    Tree.resize(4 * n);
                    build(nums, 0, n - 1, 0);
                }

                void update(int place, int val) {
                    _update(place, val, 0, n - 1, 0);
                }

                int query(int l, int r) {
                    return q(l, r, 0, n - 1, 0);
                }
            };
            ```

        * 指针实现

            ```C++
            struct SegNode {
                long long lo, hi;
                int val;
                SegNode* lchild, *rchild;
                SegNode(long long left, long long right): lo(left), hi(right), val(INT_MIN), lchild(nullptr), rchild(nullptr) {}
            };
            
            SegNode* SegSuper;
            
            SegNode* build(vector<int> arr,int left, int right) {
                SegNode* node = new SegNode(left, right);
                if (left == right) {
                    node->val = arr[left];
                    return node;
                }
                int mid = (left + right) / 2;
                node->lchild = build(arr, left, mid);
                node->rchild = build(arr, mid + 1, right);
                node->val = max(node->lchild->val,node->rchild->val);
                return node;
            }
            
            void update(SegNode* root, int place, int val) {
                if(root->lo==root->hi){
                    root->val=val;
                    return;
                }
                int mid = (root->lo + root->hi) / 2;
                if (place <= mid) {
                    update(root->lchild, place, val);
                }
                else {
                    update(root->rchild, place, val);
                }
                root->val=max(root->lchild->val,root->rchild->val);
            }
            
            int query(SegNode* root, int left, int right) const {
                if (left > root->hi || right < root->lo) {
                    return INT_MIN;
                }
                if (left <= root->lo && root->hi <= right) {
                    return root->val;
                }
                return max(query(root->lchild, left, right), query(root->rchild, left, right));
            }
            ```

* 分析

    * 时间复杂度：预处理O(nlogn) 单次查找O(logn) 单次插入O(logn)

    * 空间复杂度：O(n)

## 3. 树状数组 

### A. 单点维护 区间查询

* 细节

    维护的是数组的前缀和。

    对于一个数x，他的父节点是x+lower(x)。

    每次单点更新一个值的时候将其所有的父节点同时更新。
    
        int lowbit(int x)           		返回只保留二进制数x的最后一个1的二进制数。
        void update(int x, int d)   		单点更新x的位置增加d
        int query(int x)            		区间查询[1,x]
    
    tree下标0弃置，A下标+1。
    维护区间应该映射到[1,x+1]，x代表的是元素的个数。
    在做离散化的时候可以将其映射到[1,x+1]这样就可以不用在乎查询/插入的时候下标+1。
    
        A[1]    tree[1]=A[1];
        
        A[2]        tree[2]=A[1]+A[2];
        
        A[3]    tree[3]=A[3];
        
        A[4]            tree[4]=A[1]+A[2]+A[3]+A[4];
        
        A[5]    tree[5]=A[5];
        
        A[6]        tree[6]=A[5]+A[6];
        
        A[7]    tree[7]=A[7];
        
        A[8]                tree[8]=A[1]+A[2]+A[3]+A[4]+A[5]+A[6]+A[7]+A[8];
        
        单点更新x=1 -> 维护1 2 4 8
        
        单点更新x=2 -> 维护3 4 8
        
        区间查询x=3 -> 查询3 2 -> A[1]+A[2]+A[3]
    
* 代码实现

    ```C++
    class BIT {
    private:
        vector<int> tree;
        int n;
    
    public:
        BIT(int _n): n(_n), tree(_n + 1) {}
    
        static constexpr int lowbit(int x) {
            return x & (-x);
        }
    
        void update(int x, int d) {
            while (x <= n) {
                tree[x] += d;
                x += lowbit(x);
            }
        }
    
        int query(int x) const {
            int ans = 0;
            while (x) {
                ans += tree[x];
                x -= lowbit(x);
            }
            return ans;
        }
    
        int query(int lo, int hi) {
            return query(hi) - query(lo - 1);
        }
    };
    ```
    
* 分析

    * 时间复杂度：预处理O(nlogn) 单次查找O(logn) 单次插入O(logn)

    * 空间复杂度：O(n)

### B. 区间维护（单点维护） 区间查询

* 细节

    利用前缀和和差分的概念。
    
    先引入$A_{i} = \sum_{1}^{i} d_{x}$。
    
    那$A_{i}$的前缀和$\sum_{1}^{r} A_{i}$就等效于$\sum_{1}^{r}\sum_{1}^{i} d_{x}$。

    $\sum_{1}^{r}\sum_{1}^{i} d_{x}
    \\=d_{1}+(d_{1}+d_{2})+...+(d_{1}+d_{r})
    \\=rd_{1}+(r-1)d_{2}+...+d_{r}
    \\=\sum_{1}^{r} (r-x+1)d_{x}
    \\=r\sum_{1}^{r}{d_{x}}-\sum_{1}^{r}(x-1)d_{x}$

    所以选择维护两个BIT（$d_{x}$、$(x-1)d_{x}$）进而维护数组A的前缀和。

    单点维护可以由区间维护退变。

* 代码实现

    ```C++
    class BIT {
    private:
        vector<int> diff, diff_i;
        int n;
        void _update(int x, int d) {
            int pos = x;
            while (x <= n) {
                diff[x] += d;
                diff_i[x] += (pos - 1) * d;
                x += lowbit(x);
            }
        }
    
    public:
        BIT(int _n) : n(_n), diff(_n + 1), diff_i(_n + 1) {}
    
        static constexpr int lowbit(int x) {
            return x & (-x);
        }
    
        void update(int x, int d) {
            upate(x, x, d);
        }
    
        void update(int lo, int hi, int d) {
            _update(lo, d);
            _update(hi + 1, -d);
        }
    
        int query(int x) const {
            int res = 0, pos = x;
            while (x) {
                res += (pos * diff[x] - diff_i[x]);
                x -= lowbit(x);
            }
            return res;
        }
    
        int query(int lo, int hi) {
            return query(hi) - query(lo - 1);
        }
    };
    ```

## 4. 字典树

* 基础版

    * 基本结构

        ```C++
        struct treeNode{
            vector<treeNode*>next;
            bool isEnd;
            char cur;
            treeNode(char c){
                next.resize(26,nullptr);
                isEnd=false;
                cur=c;
            }
        };
        ```

    * 基本维护方式

        ```C++
        super=new treeNode();
        for(auto &word:words){
            treeNode *ptr=super;
            for(auto &c:word){
                if(!ptr->next[c-'a']){
                    ptr->next[c-'a']=new treeNode(c);
                }
                ptr=ptr->next[c-'a'];
            }
            ptr->isEnd=true;
        }
        ```

* AC自动机

    * 细节

        比普通的字典树再引入了fail失配指针辅助维护字典树，以方便匹配字符流。维护fail时，当当前节点是super指针，其子节点若没有指向则将其指向起点：并没有任何一个元素有这样的开始，当前字符相当于多余；若有指向则其失配指针指向super，当其失配时返回super，但其子节点仍未维护。当维护其他节点时其子节点若是没有指向则将其指向当前指针的fail指针的对应位置，若存在则其子节点的fail指针指向当前节点的fail指针的相对位置。当前位置是否存在对应的单词取决于当前位置是否存在单词以及对应fail指针指向是否存在单词。故只需要不断地根据字符移动指针就能找到是否包含这样的单词。fail指针实际上是指当前位置失配后能找到的与之前字符匹配的最长位置。

    * 基本结构

        ```C++
        struct treeNode{
            vector<treeNode*>next;
            bool isEnd;
            treeNode* fail;
            treeNode(char c){
                next.resize(26,nullptr);
                isEnd=false;
                fail=nullptr;
            }
        }
        ```

    * 基本维护方式

        ```C++
        super=new treeNode();
        for(auto &word:words){
            treeNode *ptr=super;
            for(auto &c:word){
                if(!ptr->next[c-'a']){
                    ptr->next[c-'a']=new treeNode();
                }
                ptr=ptr->next[c-'a'];
            }
            ptr->isEnd=true;
        }
        super->fail=super;
        queue<treeNode*>qu;
        for(int i=0;i<26;i++){
            if(super->next[i]){
                super->next[i]->fail=super;
                qu.push(super->next[i]);
            }
            else{
                super->next[i]=super;
            }
        }
        while(!qu.empty()){
            treeNode* cur=qu.front();
            qu.pop();
            cur->isEnd|=cur->fail->isEnd;
            for(int i=0;i<26;i++){
                if(cur->next[i]){
                    cur->next[i]->fail=cur->fail->next[i];
                    qu.push(cur->next[i]);
                }
                else{
                    cur->next[i]=cur->fail->next[i];
                }
            }
        }
        ```

* 01字典树

    解决数位问题，从高位到低位存储，完成存储与查询。多用于解决异或最大问题，反向存储，贪心地取与当前位异或为1的节点进入。

    * 代码实现

        ```C++
        class bitTree{
        public:
            bool val,end;
            vector<bitTree*>child;
            bitTree(int val):val(val),end(false),child(2,nullptr){};
        };
        ```

# **梯度下降**

多为解决k聚类问题。

## 具体步骤

  * 找到评价方程

  * 找到评价方程的偏导

  * 梯度更新

  * 向量更新

## 例题

### A. 服务中心的最佳位置

https://leetcode.cn/problems/best-position-for-a-service-centre/

* 代码实现

    ```C++
    double getMinDistSum(vector<vector<int>>& positions) {
        double eps=1e-7;
        double lr=1;
        double decay=1e-3;

        int n=positions.size();
        int batchSize=n;

        double x=0.0,y=0.0;
        for(auto &pos:positions){
            x+=pos[0];
            y+=pos[1];
        }
        x/=n;
        y/=n;

        auto cul=[&](double cx,double cy){
            double res=0;
            for(auto &pos:positions){
                res+=sqrt((pos[0]-cx)*(pos[0]-cx)+(pos[1]-cy)*(pos[1]-cy));
            }
            return res;
        };

        mt19937 gen{random_device{}()};
        
        while(true){
            shuffle(positions.begin(),positions.end(),gen);
            double px=x,py=y;
            for(int i=0;i<n;i+=batchSize){
                int m=min(i+batchSize,n);
                double dx=0.0,dy=0.0;
                for(int j=i;j<m;j++){
                    auto &pos=positions[j];
                    // +eps防止为0
                    dx+=(x-pos[0])/(sqrt((pos[0]-x)*(pos[0]-x)+(pos[1]-y)*(pos[1]-y))+eps);
                    dy+=(y-pos[1])/(sqrt((pos[0]-x)*(pos[0]-x)+(pos[1]-y)*(pos[1]-y))+eps);
                }
                x-=lr*dx;
                y-=lr*dy;
                lr*=(1-decay);
            }
            if(sqrt((x-px)*(x-px)+(y-py)*(y-py))<eps){
                break;
            }
        }
        return cul(x,y);
    }
    ```



# **常见思维**

## 1. 前缀和

一次维护，快速获取区间信息（值不可变，值可变选择线段树/树状数组）。

* 1维

    ```C++
    // 维护
    vector<int>pre(n+1);
    for(int i=0;i<n;i++){
        pre[i+1]=pre[i]+nums[i];
    }
    // 使用
    int sum_from_x_to_y=pre[y+1]-pre[x];
    ```

* 2维

    ```C++
    // 维护
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            pre[i+1][j+1]=pre[i+1][j]+pre[i][j+1]-pre[i][j]+nums[i][j];
        }
    }
    // 使用
    int sum_from_x1y1_to_x2y2=pre[x2+1][y2+1]-pre[x1][y2+1]-pre[x2+1][y1]+pre[x1][y1];
    ```

## 2. 动态规划

### 重点

* 需要保存什么状态，怎么保存状态（这里可以考虑二进制压缩状态）。

* 转移方程。 （现态==次态，这一步难求的话可以考虑记忆化深搜。）

* 边界条件。 （初始状态）

* 根据次态位置确定维护顺序，更有甚者可能结合拓扑排序确定。

* 滚动数组优化？

* 答案位置？视维护dp含义所定，过程给出\结尾给出。

    注意区间dp若需要满足非空，则考虑答案只从单一方向维护但dp正常维护。

    https://leetcode.cn/problems/max-dot-product-of-two-subsequences/

### 例子

#### A. 最大子数组和

https://leetcode.cn/problems/maximum-subarray/

* 定义dp[i]为仅考虑前i+1个数字且确定选择nums[i]的最大可能。

* 转移方程：dp[i]=max(nums[i],dp[i-1]+nums[i])

* 边界条件：dp[0]=nums[0];

* 代码实现

    ```C++
    int maxSubArray(vector<int>& nums) {
        int n=nums.size(),ret=nums[0];
        vector<int>dp(n);W
        dp[0]=nums[0];
        for(int i=1;i<n;i++){
            dp[i]=max(nums[i],dp[i-1]+nums[i]);
            ret=max(dp[i],ret);
        }
        return ret;
    }
    ```

* 注意到现态只与前一个状态相关，故可以优化。

    ```C++
    int maxSubArray(vector<int>& nums) {
        int pre=0,Maxans=nums[0];
        for(const auto &x:nums){
            pre=max(x,pre+x);
            Maxans=max(pre,Maxans);
        }
        return Maxans;
    }
    ```

#### B. 最长递增子序列问题 LIS (基于二分搜索)

https://leetcode.cn/problems/longest-increasing-subsequence/

* 关键点

    维护一个len数组，len[i]代表长度为i的子序列最后一位的最小可能值。由于len[i]一定是单调递增的（否则当前len并不是最优的。），故维护时可以利用二分查找减少时间复杂度。这里pos维护的是上一个可行的位置，l==r时可能l也是可行位置故需要再次判断，且能确保len[pos+1]>=nums[i]。

* 代码实现

    ```C++
    int lengthOfLIS(vector<int>& nums) {
        int maxlen=1,n=nums.size();
        vector<int>len(n+1);
        len[maxlen]=nums[0];
        for(int i=1;i<n;i++){
            if(nums[i]>len[maxlen]){
                len[++maxlen]=nums[i];
            }
            else{
                int l=1,r=maxlen,pos=0;
                while(l<=r){
                    int mid=(l+r)>>1;
                    if(len[mid]<nums[i]){
                        pos=mid;
                        l=mid+1;
                    }
                    else{
                        r=mid-1;
                    }
                }
                len[pos+1]=nums[i];
            }
        }
        return maxlen;
    }
    ```

    ```C++
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        vector<int>len;
        len.push_back(nums[0]);
        for(int i=1;i<n;i++){
            if(nums[i]>len.back()){
                len.push_back(nums[i]);
            }
            else{
                auto it=lower_bound(len.begin(),len.end(),nums[i]);
                *it=nums[i];
            }
        }
        return len.size();
    }
    ```

* 二维版本

    https://leetcode.cn/problems/delete-columns-to-make-sorted-iii/

    * 代码实现

        ```C++
        int minDeletionSize(vector<string>& strs) {
            int n=strs[0].size(),m=strs.size();
            vector<int>dp(n,1);
            auto bigger=[&](int pre,int cur){
                for(int i=0;i<m;i++){
                    if(strs[i][pre]>strs[i][cur])return false;
                }
                return true;
            };
            for(int i=1;i<n;i++){
                for(int j=0;j<i;j++){
                    if(bigger(j,i))dp[i]=max(dp[i],dp[j]+1);
                }
            }
            return n-*max_element(dp.begin(),dp.end());
        }
        ```

#### C. 背包问题

##### I.完全背包问题

###### a. 面试题 08.11. 硬币

无穷物品问题。

https://leetcode.cn/problems/coin-lcci/

对每一个物品，在每一种可能的条件下维护它，而不是在每一种可能的条件维护每一个物品。

* 代码实现

    ```C++
    int mod=1e9+7;
    int coins[4]={1,5,10,25};
    int waysToChange(int n) {
        //dp[i]=dp[i-1]+dp[i-5]+dp[i-10]+dp[i-25]会产生重复 1-5和5-1就会重复
        //每次只考虑一种硬币coin，dp[i]+=dp[i-coins]并计算到底,每次计算都是基于之前的选择
        //dp[0]=1,ans=dp[n]
        vector<int>dp(n+1);
        dp[0]=1;
        for(auto coin:coins){
            for(int i=coin;i<=n;i++){
                dp[i]=(long long)(dp[i]+dp[i-coin])%mod;
            }
        }
        return dp[n];
    }
    ```

###### b. 多项式乘法问题

* $(1+x+x^2+...)(1+x^2+x^4)...(1+x^n+x^2n)... = res*x^n$ 问题

    将一个物品的权重抽象为一个多项式中的指数，这样可以变为一个物品无限的背包问题，每次考虑将新有的物品加入到之前可能存在的集合当中，奉献新的结果。如此处的$(1+x+x^2+...)(1+x^2+x^4)...(1+x^n+x^2n)... = res*x^n$就将权重为1的物品变为了$(1+x+x^2+...)$分别是不选、选一个、选多个。

    对此问题的多项式进行修改可以达成多个不同权重的无限物品最终选取权重的可能数。

    * 代码实现

        ```C++
        int c1[10010],c2[10010];
        ll getVal(ll n,ll mod){
            /* (1+x+x^2+...)(1+x^2+x^4)...(1+x^n+x^2n)... = res*x^n */
            for(int i=0;i<=n;i++){
                c1[i]=1,c2[i]=0;
            }
            //物品权重
            for(int i=2;i<=n;i++){
                //开始端点
                for(int j=0;j<=n;j++){
                    //物品选择个数
                    for(int k=0;j+k<=n;k+=i){
                        c2[j+k]=(c2[j+k]+c1[j])%mod;
                    }
                }
                //滚动优化
                for(int j=0;j<=n;j++){
                    c1[j]=c2[j],c2[j]=0;
                }
            }
            return c1[n];
        }
        ```

    * 更可以优化为

        ```C++
        int dp[10010];
        ll getVal(ll n,ll mod){
            /* (1+x+x^2+...)(1+x^2+x^4)...(1+x^n+x^2n)... = res*x^n */
            for(int i=0;i<=n;i++){
                dp[i]=1;
            }
            for(int i=2;i<=n;i++){
                for(int j=i;j<=n;j++){
                    dp[j]=(dp[j-i]+dp[j])%mod;
                }
            }
            return dp[n];
        }
        ```

##### II. 0-1背包问题

* 细节

    0-1背包问题意思是指每一个物品只有取\不取两种可能。而处理此类问题，通常有一个限定的上界，在这个上界限制下求价值和最大。这里的上界不代表其真的用完，只是一个边界。


* 例子

    * 物品价值最大问题

        给定$n$个物品和$1$个背包，以及$n$个物品的信息：重量$w_i$、体积$b_i$、价值$v_i$，背包的最大容量$c$，背包的最大容积$d$。求携带的物品的最大价值。

        * 细节

            对于每个物品，都只有取或者不取两种方案。

            $dp[i][c][d]=\left\{\begin{matrix} &dp[i+1][c][d] &c<w_i||d<b_i \\
            &\max(dp[i+1][c-w_i][d-b_i]+v_i,dp[i+1][c][d]) &other \end{matrix}\right. $
            
            $dp[i][c][d]$代指考虑$[i,n)$下标的元素、当前剩余容量为c、剩余容积为d的问题的解。

            注意选不上时只能将状态迁移到下一个子问题。

        * 代码实现

            ```C++
            vector<vector<vector<int>>>dp(n+1,vector<vector<int>>(tw+1,vector<int>(tb+1)));
            for(int i=n-1;i>=0;i--){
                for(int j=0;j<w[i];j++){
                    for(int k=b[i];k<=tb;k++){
                        dp[i][j][k]=dp[i+1][j][k];
                    }
                }
                for(int j=w[i];j<=tw;j++){
                    for(int k=0;k<b[i];k++){
                        dp[i][j][k]=dp[i+1][j][k];
                    }
                    for(int k=b[i];k<=tb;k++){
                        dp[i][j][k]=max(dp[i+1][j-w[i]][k-b[i]]+v[i],dp[i+1][j][k]);
                    }
                }
            }
            ```

        * 路径还原

            根据当前状态是由哪一个状态迁移来的确定。

#### D. 树节点的第 K 个祖先（倍增DP）

* 细节

    利用$dp[i][j]$代表i节点的第$2^j$个祖先节点

        1. 显然dp[i][0]=parent[i]
        2. 利用性质2^(j-1)+2^(j-1)=2^j
        3. 知dp[i][j]=dp[dp[i][j-1]][j-1]

    寻找第k个祖先节点时：

        1. 首先考虑将k分解成∑2^i 
        2. 优先低位，依次考虑

* 代码实现

    ```C++
    class TreeAncestor {
    public:
        vector<vector<int>>dp;
        TreeAncestor(int n, vector<int>& parent) {
            dp.resize(n);
            for(int i=0;i<n;i++){
                dp[i].push_back(parent[i]);
            }
            int j=1;
            while(true){
                bool none=true;
                for(int i=0;i<n;i++){
                    int cur=dp[i][j-1]!=-1?dp[dp[i][j-1]][j-1]:-1;
                    dp[i].push_back(cur);
                    if(cur!=-1)none=false;
                }
                j++;
                if(none)break;
            }
        }
        
        // 递归
        // int getKthAncestor(int node, int k) {
        //     if(k==0||node==-1)return node;
        //     int pos=getRightOne(k);
        //     return pos<dp[node].size()?getKthAncestor(dp[node][pos],k-(1<<pos)):-1;
        // }
    
        // int getRightOne(int num){
        //     num&=(~num+1);
        //     int pos=-1;
        //     while(num){
        //         num>>=1;
        //         pos++;
        //     }
        //     return pos;
        // }
    
        // 递推
        int getKthAncestor(int node, int k) {
            int ret=node,pos=0;
            while(k&&ret!=-1){
                if(pos>=dp[ret].size())return -1;
                if(k&1)ret=dp[ret][pos];
                pos++;
                k>>=1;
            }
            return ret;
        }
    };
    ```

* 分析

    * 时间复杂度:O(nlogn)

#### E. 最长公共子序列

* 代码实现

    ```C++
    string longestSubString(string str1, string str2) {
        int n1=str1.size(),n2=str2.size();
        vector<vector<int>>dp(n1+1,vector<int>(n2+1));
        for(int i=1;i<=n1;i++){
            for(int j=1;j<=n2;j++){
                if(str1[i-1]==str2[j-1])dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
        return dp[n1][n2];
    }
    ```

* 应用

    最短公共超序列

    https://leetcode.cn/problems/shortest-common-supersequence/

    ```C++
    string shortestCommonSupersequence(string str1, string str2) {
        int n1=str1.size(),n2=str2.size();
        vector<vector<int>>dp(n1+1,vector<int>(n2+1));
        for(int i=1;i<=n1;i++){
            for(int j=1;j<=n2;j++){
                if(str1[i-1]==str2[j-1])dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
        string sub;
        int ptr1=n1,ptr2=n2;
        while(ptr1>0&&ptr2>0){
            if(str1[ptr1-1]==str2[ptr2-1]){
                sub+=str1[ptr1-1];
                ptr1--,ptr2--;
            }
            else if(dp[ptr1][ptr2]==dp[ptr1][ptr2-1]){
                sub+=str2[ptr2-1];
                ptr2--;
            }
            else {
                sub+=str1[ptr1-1];
                ptr1--;
            }
        }
        while(ptr1)sub+=str1[--ptr1];
        while(ptr2)sub+=str2[--ptr2];
        reverse(sub.begin(),sub.end());
        return sub;
    }
    ```

#### F. 回文串问题

* 细节

    $is[i][j]$代指字符串[i:j]是否是一个回文串。考虑中心出发：$is[i][j]=true$的条件是$s[i]==[j]$&&$(j-i<=1||dp[i+1][j-1])$。由于转移方程的特性，i自底而下，j自左而右。

* 代码实现

    ```C++
    int getPalindromicInfo(string s) {
        int n=s.size();
        vector<vector<bool>>is(n,vector<bool>(n));
        for(int i=n-1;i>=0;i--){
            for(int j=i;j<n;j++){
                if(s[i]==s[j]){
                    if(j-i<=1||is[i+1][j-1]){
                        is[i][j]=true;
                    }
                }
            }
        }
        return is;
    }
    ```

* 分析

    * 时间复杂度：O($n^2$)
    
    * 空间复杂度：O($n^2$)

#### G. 分组问题

将一个集合分为大小相等的两个不重叠子集。

* 细节

    当我们依次遍历集合元素时，将他放在多的一个集合时记为val，将他放在少的一个集合时记为-val，将其丢弃时取为0。将差值的绝对值d作为key，将最小高度h作为value。每次我们先将前一个状态的情况（没有加这个元素之前）存起来，逐个遍历前一个状态的情况，故选取当前元素val放在多的一个集合时就有$dp[d+val]=max(dp[d+val],h)$，放在少的集合就有$dp[abs(d-val)]=max(dp[abs(d-val)],h+min(val,d))$（abs将总是指向高的和矮的差值）。

* 代码实现

```C++
int biggestTwoSubSum(vector<int>& rods) {
    int n=rods.size();
    map<int,int>dp;
    dp[0]=0;
    for(auto &val:rods){
        map<int,int>pre(dp);
        for(auto &[d,h]:pre){
            dp[val+d]=max(dp[val+d],h);
            dp[abs(d-val)]=max(dp[abs(d-val)],h+min(val,d));
        }
    }
    return dp[0];
}
```

* 分析

    * 时间复杂度：O($nS$)
    
    * 空间复杂度：O($nS$)

#### H. 博弈问题

* 细节

    考虑当前玩家在当前状态是否能获胜。

* 例子

    * 探求两个玩家都最优操作的赢家，利用每个人都希望自己的分数最大的特点，当前得分减去另一个玩家上一次的分数的最大来求解。一般从后往前走，最后谁先手都无所谓，因为是对称的。维护dp[i]是从i之后的元素中先手且最优的得分。

        https://leetcode.cn/problems/stone-game-iii/

    * 对于当前玩家，若当前操作可以导向必败态则当前玩家必胜。每次只考虑当前玩家的情况，而将其他子问题看作是另一个玩家。

        https://leetcode.cn/problems/stone-game-iv/submissions/

#### I. 最大子段和问题

* 细节

    使用$dp[i]$代指以$nums[i]$结尾的最大子段。而在考虑将$nums[i]$加入时考虑与之前的连接或自成一段。

    $ dp[i]=\max(dp[i-1]+nums[i],nums[i]) $

    $ res=\max(dp[i]) $

* 代码实现

    ```C++
    int maxSubSum(vector<int>&nums){
        int dp=0,res=0;
        for(auto &num:nums){
            dp=max(dp+num,num);
            res=max(res,dp);
        }
        return res;
    }
    ```

* 分析

    * 时间复杂度：O($n$)

    * 空间复杂度：O($1$)

#### J. 最大子矩阵和问题

* 细节

    $$sum[i_1][j_1][i_2][j_2]=\sum_{i=i_1}^{i_2}\sum_{j=j_1}^{j_2}matrix[i][j]$$
    
    $$
    \begin{aligned}
    res&=\max_{1<=i_1<=i_2<=n}\max_{1<=j_1<=j_2<=m}sum[i_1][j_1][i_2][j_2]\\
    &=\max_{1<=i_1<=i_2<=n}t[i_1][i_2]
    \end{aligned}
    $$

    $$
    \begin{aligned}
    t[i_1][i_2]&=\max_{1<=j_1<=j_2<=m}sum[i_1][j_1][i_2][j_2]\\
    &=\max_{1<=j_1<=j_2<=m}\sum_{j=j_1}^{j_2}\sum_{i=i_1}^{i_2}matrix[i][j]\\
    &=\max_{1<=j_1<=j_2<=m}\sum_{j=j_1}^{j_2}b[j]
    \end{aligned}
    $$

    $$b[j]=\sum_{i=i_1}^{i_2}matrix[i][j]$$

    这样就将问题转化为求$b[j]$最大子段和的一个过程。枚举行->选择最大列。

* 代码实现

```C++

int maxSubSum(vector<int>&nums){
    int dp=0,res=0;
    for(auto &num:nums){
        dp=max(dp+num,num);
        res=max(res,dp);
    }
    return res;
}

int maxSubSum2(vector<vector<int>>&matrix){
    int res=0;
    int n=matrix.size(),m=matrix[0].size();
    for(int i1=0;i1<n;i1++){
        vector<int>b(m);
        for(int i2=i1;i2<n;i2++){
            for(int k=0;k<m;k++){
                b[k]+=matrix[i2][k];
            }
            int maxb=maxSubSum(b);
            res=max(maxb,res);
        }
    }
    return res;
}
```

#### K. 流水作业问题

* 细节

    当前有机器$M_1、M_2$，N个作业$\{1,2,3,...,N\}$需要依次经过这两个机器，且其依次需要消耗$A_i、B_i$时间。最好的选择是让$M_1$不断工作，让$M_2$等待或处理。

    假设$T(S,t)$为规模为$S$的作业群需要被加工，此时$M_2$需要等待$t$后才能利用，完成任务群的最短时间。故问题的答案是$T(S,0)$，最终$T(0,t)=t$。

    有等式
    
    $$T(S,t)=\min_{i \in S}\{T(S-\{i\},B_i+\max(t-A_i,0)\}$$

    假设在最优序列中i任务第一个执行，j任务第二个执行。

    故有

    $$
    \begin{aligned}
    T(S,t)&=A_i+T(S-\{i\},B_i+\max(t-A_i,0))
    \\&=A_i+A_j+T(S-\{i,j\},B_j+\max(B_i+\max(t-A_i,0)-A_j,0))
    \end{aligned}
    $$

    又设
    $$
    \begin{aligned}
    t_{ij}&=B_j+\max(B_i+\max(t-A_i,0)-A_j,0)
    \\&=B_j+B_i-A_j+\max(\max(t-A_i,0),A_j-B_i)
    \\&=B_j+B_i-A_j+\max(t-A_i,0,A_j-B_i)
    \\&=B_j+B_i-A_j-A_i+\max(t,A_i,A_j+A_i-B_i)
    \end{aligned}
    $$

    $t_{i,j}受限于\max(t,A_i,A_j+A_i-B_i)$

    称$\min(B_i,A_j)>=\min(B_j,A_i)$为Johnson不等式。

    满足该不等式时

    $$
    \begin{aligned}
    &\min(B_i,A_j)>=\min(B_j,A_i)
    \\&\Rightarrow \max(-B_i,-A_j)<=\max(-B_j,-A_i)
    \\&\Rightarrow A_i+A_j+\max(-B_i,-A_j)<=A_i+A_j+\max(-B_j,-A_i)
    \\&\Rightarrow \max(A_i+A_j-B_i,A_i)<=\max(A_i+A_j-B_j,A_j)
    \\&\Rightarrow \max(A_i+A_j-B_i,A_i,t)<=\max(A_i+A_j-B_j,A_j,t)
    \end{aligned}
    $$

    故可以知道，$t_{i,j}<=t_{j,i}$。

    故当每一个任务都满足Johnson不等式时，一定是最优。

    我们将这些工作分为$N_1=\{i|A_i<B_i\}$，$N_2=\{i|A_i>=B_i\}$。将$N_1$按$A_i$递增排序，$N_2$按$B_i$递减排序，以$N_1$+$N_2$作为答案就是最优的。由于$N_1$中$A_i$递增，所以$i<j$ 时$Ai<=A_j<B_i$故有$\min(B_i,A_j)=A_j>=\min(B_j,A_i)$恒成立；对于$N_2$中$B_i$递减，所以$i<j$ 时$A_i>=B_i>=B_j$故有$\min(B_i,A_j)>=\min(B_j,A_i)=B_j$恒成立；对于$N_1$的最后一个元素和$N_2$的最前一个元素来说，当$A_i<=B_j$时$A_i<=B_j<=A_j与A_i<B_i$故有$\min(B_i,A_j)>=\min(B_j,A_i)=A_i$恒成立，反之也是一致。

    由此说明了正确性。

* 代码实现

    ```C++
    int FlowShop(vector<int>&A,vector<int>&B){
        int n=A.size();
        vector<int>N1,N2;
        for(int i=0;i<n;i++){
            if(A[i]<B[i]){
                N1.push_back(i);
            }
            else{
                N2.push_back(i);
            }
        }
        sort(N1.begin(),N1.end(),[&](int &idx1,int &idx2){
            return A[idx1]<A[idx2];
        });
        sort(N2.begin(),N2.end(),[&](int &idx1,int &idx2){
            return B[idx1]>B[idx2];
        });
        N1.insert(N1.end(),N2.begin(),N2.end());
        int M1t=A[N1[0]],M2t=A[N1[0]]+B[N1[0]];
        for(int i=1;i<n;i++){
            M1t+=A[N1[i]];
            M2t=max(M1t,M2t)+B[N1[i]];
        }
        return M2t;
    }
    ```

* 分析

    * 时间复杂度：O($nlogn$)
    
    * 空间复杂度：O($n$)

#### L. 数位DP

* 细节

    对一个求边界之内的问题可以考虑数位DP。使用$dp[pos][stats][bound]$来表示当前求解到$pos$位置，而$stats$根据不同问题有不同的取法，但$bound$相对固定：一般以$bound=0$来代指当前的状态与上下界没有关系可以任意取（最小->最大）、$bound=1$来代指当前的状态贴着下界（下界->最大）、$bound=2$来代指当前的状态贴着上界（最小-上界）、$bound=3$代指当前状态贴着上下界（下界-上界）。

    $bound=(00)_b$  可选范围（最小->最大）

    $bound=(01)_b$ 可选范围（下界->最大）

    $bound=(10)_b$ 可选范围（最小-上界）

    $bound=(11)_b$ 可选范围（下界-上界）

    低位的1代表当前是否与下界相贴，高位的1代表当前是否与上界相贴。

    故可以有以下操作：

    ```C++
    l=(bound&1?low[pos]:LOW);
    r=(bound&2?up[pos]:UP);

    next_bound=(bound&1?cur==low[pos]:0)|(bound&2?(cur==up[pos])<<1:0);
    ```

    $stats$的维护依据题目而定。

* 例子

    * 找到所有好字符串

        https://leetcode.cn/problems/find-all-good-strings/

        * 代码实现

            ```C++
            class Solution {
            public:
                int findGoodStrings(int n, string s1, string s2, string evil) {
                    int m=evil.size(),mod=1e9+7;
                    
                    //每次失配时eptr要按next转跳
            
                    auto get_next=[&](string _str)->vector<int> {
                        int _n=_str.size(),_ptr=0,_prek=-1;
                        vector<int>_ret(_n+1,-1);
                        while(_ptr<_n){
                            if(_prek==-1||_str[_ptr]==_str[_prek]){
                                _ret[++_ptr]=++_prek;
                            }
                            else{
                                _prek=_ret[_prek]; 
                            }
                        }
                        return _ret;
                    };
                    vector<int>next(move(get_next(evil)));
            
                    vector<vector<int>>nextmem(m,vector<int>(26,-1));
                    auto getNext=[&](int ptr,char cur){
                        if(nextmem[ptr][cur-'a']!=-1)return nextmem[ptr][cur-'a'];
                        if(evil[ptr]==cur)return ptr+1;
                        int prek=ptr;
                        while(prek!=-1&&evil[prek]!=cur){
                            prek=next[prek];
                        }
                        return nextmem[ptr][cur-'a']=prek+1;
                    };
            
                    vector<vector<vector<int>>>dp(n,vector<vector<int>>(m,vector<int>(4,-1)));
                    function<int(int,int,int)> getDp=[&](int ptr,int eptr,int bound){
                        if(eptr==m)return 0;//匹配完全（子串存在evil）
                        else if(ptr==n)return 1;
                        else if(dp[ptr][eptr][bound]!=-1)return dp[ptr][eptr][bound];
            
                        dp[ptr][eptr][bound]=0;
                        char l=(bound&1?s1[ptr]:'a');
                        char r=(bound&2?s2[ptr]:'z');
                        for(char ch=l;ch<=r;ch++){
                            int next_eptr=getNext(eptr,ch);
                            int next_bound=(bound&1?ch==s1[ptr]:0)|(bound&2?(ch==s2[ptr])<<1:0);
                            dp[ptr][eptr][bound]+=getDp(ptr+1,next_eptr,next_bound);
                            dp[ptr][eptr][bound]%=mod;
                        }
                        return dp[ptr][eptr][bound];
                    };
            
                    return getDp(0,0,3);
                }
            };
            ```

#### M. 最优搜索二叉树

关注优化方法。

* 细节

    假设给定一个二叉树的搜索概率，求解出最优的二叉树结构。搜索二叉树的内点由存在的元素构成，而叶子节点由匹配失败区间组成。如下示意图。

                     5
                  /     \
               2           6
             /   \       /   \
        (-inf,2) (2,5) (5,6) (6,inf)

    对于每个结点无论树的结构怎么变，其值都应该是一致的。假设将这些存在的元素标记为$S=\{x_0,x_1,x_2,x_3,...,x_n\}$,而其中的元素之间的间隙就是叶子结点，已知$a_i=p_{x=(x_{i-1},x_i)}$，$b_i=p_{x=x_i}$。

    设定$w_{i,j}=a_{i-1}+b_i+...+b_j+a_j$，由于$n$个元素有$n+1$个区间，所以$a$会比$b$多一个元素。又设定$p_{i,j}$为到达此子树的平均路径。

    故可以有$w_{i,j}(p_{i,j}-1)=w_{i,m-1}p_{i,m-1}+w_{m+1,j}p_{m+1,j}$。一整颗子树的权重是$w_{i,j}$，去除根节点后剩下两颗子树，其产生的开销等于去除到父节点的边的$p_{i,j}$乘以总权重。

    故有$w_{i,j}p_{i,j}=w_{i,j}+w_{i,m-1}p_{i,m-1}+w_{m+1,j}p_{m+1,j}$。

    假设$dp[i][j]=w_{i,j}p_{i,j}$。
    
    有转移方程$dp[i][j]=w[i][j]+min_{i<=k<=j}\{dp[i][k-1]+dp[k+1][j]\}$。$j<i,dp[i][j]=0$。

    直接枚举根节点k的时间复杂度显然是$O(n^3)$。
    
    但显然可以有更优的解法，注意到$dp[i][j]$的两个边界迁移状态是$dp[i][j-1]$、$dp[i+1][j]$，显然$dp[i][j]$与这两个状态相差两端。而对于$dp[i][j-1]$其加入右端后其根节点倾向一定是向右偏以平衡、而$dp[i+1][j]$则相反。只需要检查下图$O$对应位置即可。

        i                 j-1
        [======X===========]
         i+1                 j
          [============X=====]
        i                    j  
        [======OOOOOOOOO=====]  

    即有：$dp[i][j]=w[i][j]+min_{root[i][j-1]<=k<=root[i+1][j]}\{dp[i][k-1]+dp[k+1][j]\}$。$j<i,dp[i][j]=0$。此时期望复杂度为$O(n^2)$

* 代码实现

    ```C++
    /* 预处理w[i][j] */
    // 注意将dp[i][i-1]视作了一子树为空的情况。
    vector<vector<int>>root(n,vector<int>(n)),dp(n,vector<int>(n));
    for(int i=n;i>0;i--){
        for(int j=i;j<=n;j++){
            int l=max(root[i][j-1],i),r=min(root[i+1][j],j);
            root[i][j]=l;
            dp[i][j]=dp[i][l-1]+dp[l+1][j];
            for(int k=l+1;k<=r;k++){
                int temp=dp[i][k-1]+dp[k+1][j];
                //等于也划入（减少之后的开销）
                if(temp<=dp[i][j]){
                    dp[i][j]=temp;
                    root[i][j]=k;
                }
            }
            dp[i][j]+=w[i][j];
        }
    }
    ```

#### N. 轮廓线优化

* 例子

    https://leetcode.cn/problems/maximize-grid-happiness/submissions/

    * 细节

        假设按顺序考虑每个格子，那么可以得出对于每个格子，其受其上方、左边的格子影响（每个格子都会被考虑一次）。我们可以保留前n个元素的mask，完成对下一个状态的迭代。

    * 代码实现

        ```C++
        class Solution {
        public:
            int getMaxGridHappiness(int m, int n, int introvertsCount, int extrovertsCount) {
                int _n=pow(3,n);
                vector<vector<vector<vector<int>>>>dp(m*n+1,vector<vector<vector<int>>>(introvertsCount+1,vector<vector<int>>(extrovertsCount+1,vector<int>(_n,0x80f0f0f0))));
                dp[0][0][0][0]=0;
                int score[3][3]={{0,0,0},{0,-60,-10},{0,-10,40}};
                for(int pos=0;pos<m*n;pos++){
                    for(int mask=0;mask<_n;mask++){
                        int left=mask%3,up=mask*3/_n;
                        if(pos%n==0){
                            left=0;
                        }
                        for(int in=0;in<=introvertsCount;in++){
                            for(int ex=0;ex<=extrovertsCount;ex++){
                                int curbit=mask%3,cur=dp[pos][in][ex][mask];
                                //in
                                if(in<introvertsCount){
                                    dp[pos+1][in+1][ex][(mask*3+1)%_n]=max(dp[pos+1][in+1][ex][(mask*3+1)%_n],cur+120+score[left][1]+score[up][1]);
                                }
                                //ex
                                if(ex<extrovertsCount){
                                    dp[pos+1][in][ex+1][(mask*3+2)%_n]=max(dp[pos+1][in][ex+1][(mask*3+2)%_n],cur+40+score[left][2]+score[up][2]);
                                }
                                //none
                                dp[pos+1][in][ex][(mask*3)%_n]=max(dp[pos+1][in][ex][(mask*3)%_n],cur);
                            }
                        }
                    }
                }
                int res=0;
                for(int i=0;i<=introvertsCount;i++){
                    for(int j=0;j<=extrovertsCount;j++){
                        for(int mask=0;mask<_n;mask++){
                            res=max(res,dp[m*n][i][j][mask]);
                        }
                    }
                }
                return res;        
            }
        };
        ```

* 分析

    * 时间复杂度：$O(n^2)$

    * 空间复杂度：$O(n^2)$

#### O. 插入元素DP

从空白/特定情况出发，逐步插入元素的最优问题。如下例。

* 细节

    每次向数组中取出除头尾的一个元素，其开销为其与左右的乘积。求只剩下2个元素时最小开销。

    $10,1,20,5 \to 10,1,20,5 \to 10,1,5 \to 10,5$的开销为$1150$。

    正向考虑麻烦且复杂，考虑使用动态规划自底向上计算。

    问题与左右边界高度相关，不妨设$dp[i][j]$为$i,j$位置定的情况下最优开销。将中间位置$k$插入后还要考虑将$[i,k]$、$[k,j]$的开销。

    $dp[i][j]=min(dp[i][j],dp[i][k]+dp[k][j]+v[i]*v[k]*v[j])$

    注意边界条件的处理。

* 代码实现

    ```C++
    vector<vector<ll>>dp(n,vector<ll>(n,INF));
	for(int i=0;i<n;i++){
		dp[i][i]=0;
	}
	for(int i=0;i+1<n;i++){
		dp[i][i+1]=0;
	}
	for(int i=n-1;i>=0;i--){
		for(int j=i;j<n;j++){
			for(int k=i+1;k<j;k++){
				dp[i][j]=min(dp[i][j],dp[i][k]+dp[k][j]+v[k]*v[i]*v[j]);
			}
		}

	}
    ```

* 分析

    * 时间复杂度：$O(n^3)$

    * 空间复杂度：$O(n^2)$

## 3. 二分搜索

解决寻找单调函数零点问题/寻找单调函数最大符合点

### A. 结合贪心算法找到最大符合点

check函数的维护。

### B. 基于二分查找的动态规划

## 4. 拓扑排序

解决事件先后发生关系。用单向边依次连接各事件，找到入度为0的点作为起点，再不断加入新的入度为0的点，直到完成遍历。

## 5. 贪心

### 例子

#### A. Boyer-Moore 投票算法

筛选大于半数的元素

https://leetcode.cn/problems/majority-element/

* 代码实现

    ```C++
    int majorityElement(vector<int>& nums) {
        int candidate = -1;
        int count = 0;
        for (int num : nums) {
            if (num == candidate)
                ++count;
            else if (--count < 0) {
                candidate = num;
                count = 1;
            }
        }
        return candidate;
    }
    ```

#### B. 区间处理-选取最少区间元素覆盖指定区间

* 细节

    元素按l从小到大排序，l相同时r从大到小。依次遍历，维护上一次r，每次在可选范围内选取能到最大新r的元素。

* 代码实现

    ```C++
    //need[0,n]
    sort(range.begin(),range.end(),[&](vector<int>&v1,vector<int>&v2){
        if(v1[0]==v2[0])return v1[1]>v2[1];
        return v1[0]<v2[0];
    });
    int ptr=0,res=0,r=0,_n=range.size();
    while(ptr<_n&&r<n){
        int bound=r;
        while(ptr<_n&&range[ptr][0]<=r){
            bound=max(bound,range[ptr++][1]);
        }
        if(bound>r){
            r=bound;
            res++;
        }
        else return -1;
    }
    return r>=n?res:-1;
    ```

#### C. 区间处理-选取最少的点使得所有的区间都至少在范围里有一个点

* 细节

    元素按l从小到大排序，r无所谓。依次遍历，维护上一次的r，与当前的r取最小，当和上一次的区间不重叠时，加入新的点。

* 代码实现

    ```C++
    sort(range.begin(),range.end(),[&](vector<int>&v1,vector<int>&v2){
        if(v1[0]==v2[0])return v1[1]>v2[1];
        return v1[0]<v2[0];
    });
    int ptr=0,res=0,prer=-1,n=range.size();
    while(ptr<n){
        if(prer>=range[ptr][0]){
            prer=min(prer,range[ptr][1]);
        }
        else{
            prer=range[ptr][1];
            res++;
        }
        ptr++;
    }
    return res;
    ```


#### D. Nim问题

* 问题阐述

    Nim问题是一种经典的博弈论问题，它的规则是这样的：有若干堆石子，每堆石子的数量都是有限的，两个玩家轮流从其中的一堆取石子，每次至少取一颗，最多可以取完当前堆，无法继续取石子的玩家输掉游戏。

* 细节

    这个问题可以化简为异或是否为0的问题，当所有元素异或为0时为必败态，否则为必赢态。证明如下：
    
    1. 当所有堆都为0时显然是一个必败态。
    2. 假设当所有小于n个石子的局面都成立。
    3. 对于任意一个大于n的局面且异或值不为0的状态：我们要找到该玩家的一个合法操作使另一个玩家变成必败态，而必败态无论如何操作都会令另一玩家处于必胜态。而前者可以通过寻找当前异或值的最高位1并找到这个值，将该值变化为导致异或为0。

    故可知：当异或值Nim和为0时玩家必败，非零时必胜。

#### E. 阶梯Nim问题

* 问题阐述

    石子在若干个阶梯上，玩家每次只能将一层的若干石子搬到下一层，直到全部石子都搬到了第0层。

* 细节

    * 偶数阶不影响问题的结果
    
    * 对奇数阶异或为0为必败态    

## 6. 数组处理

### 例子

#### A. 获取k位最大字典序子序列

* 细节

    应用单调栈的思想，将尽量大的数字放在前面->单调递减的栈。但最大抛弃数字是n-size，当不能抛弃数字的时候注意不进行单调处理。

* 代码实现

    ```C++
    auto getMaxSub=[&](vector<int>&nums,int size)->vector<int>{
        vector<int>ret(size);
        int _n=nums.size();
        int top=-1,cnt=_n-size;
        for(int i=0;i<_n;i++){
            while(top>=0&&ret[top]<nums[i]&&cnt>0){
                top--;
                cnt--;
            }
            if(top<size-1){
                ret[++top]=nums[i];
            }
            else{
                cnt--;
            }
        }
        return ret;
    };
    ```

* 分析

    * 时间复杂度：O(n)

    * 空间复杂度：O(n)

#### B. 合并两个数组获得最大字典序

字符串也有对应的操作。

* 分析

    维护两个指针ptr1、ptr2，对应当前两个数组最前端位置。nums1[ptr1]!=nums2[ptr2]时，选择最大的一个是最优的；但当nums1[ptr1]==nums2[ptr2]时，需要继续比对，看哪一边能先取得最大的，这就导致了大的能更快被取到。

* 代码实现

    ```C++
    auto compareArrByStartIndex=[&](vector<int>&v1,int idx1,vector<int>&v2,int idx2)->int{
        int n1=v1.size(),n2=v2.size();
        while(idx1<n1&&idx2<n2){
            int diff=v1[idx1]-v2[idx2];
            if(diff!=0)return diff;
            idx1++,idx2++;
        }
        return (n1-idx1)-(n2-idx2);
    };
    auto merge2=[&](vector<int>&v1,vector<int>&v2)->vector<int>{
        int n1=v1.size(),n2=v2.size();
        if(!n1)return v2;
        else if(!n2)return v1;
        vector<int>ret(n1+n2);
        int ptr1=0,ptr2=0;
        for(int i=0;i<n1+n2;i++){
            if(compareArrByStartIndex(v1,ptr1,v2,ptr2)>0){
                ret[i]=v1[ptr1++];
            }
            else{
                ret[i]=v2[ptr2++];
            }
        }
        return ret;
    };
    ```

* 分析

    * 时间复杂度：O($n_1n_2$)

    * 空间复杂度：O($n_1n_2$)

#### C. 满足一定条件的连续子数组

* 当连续子数组要求的性质具有一定的单调性，则使用滑动窗口。

* 使用记录前缀的方式来维护数目。

#### D. 最小的第k个数

* 细节

    将数组分为多个大小为5的元素，对其排序后得到中位数，再找到中位数的中位数，此时至少可以排除30%的答案，再进入子问题。

* 代码实现

    ```C++
    int Partition(vector<int>&nums,int x){
        int n=nums.size();
        for(int i=0;i<n;i++){
            if(nums[i]==x){
                swap(nums[i],nums[0]);
            }
        }
        int l=0,r=nums.size()-1;
        while(l<r){
            while(l<r&&nums[r]>x){
                r--;
            }
            nums[l]=nums[r];
            while(l<r&&nums[l]<x){
                l++;
            }
            nums[r]=nums[l];
        }
        nums[l]=x;
        return l;
    }
    
    int PickNumK(vector<int>&nums,int k,int l,int r){
        if(r-l<75){
            sort(nums.begin()+l,nums.begin()+r);
            return nums[l+k-1];
        }
        for(int i=0;i<=(r-k-4)/5;i++){
            sort(nums.begin()+l+i*5,nums.begin()+l+i*5+4);
            swap(nums[l+i*5+2],nums[l+i]);
        }
        int x=PickNumK(nums,(r-l+6)/10,l,l+(r-l-4)/5);
        int idx=Partition(nums,x),size=idx-l+1;
        if(size>=k){
            return PickNumK(nums,k,l,idx);
        }
        return PickNumK(nums,k-size,idx+1,r);
    }
    ```

#### D. 逆数对问题

将一个数组中$nums[i]>nums[j]$且$i<j$的一对$(i,j)$叫做逆数对。

这类问题可以抽象为冒泡排序的比较次数计算、将元素依次移动的开销计算。

* 细节

    对于每一个元素，其需要计算其左边的元素有多少个比他大，朴素的思想为$O(n^2)$。但我们考虑到每次和左边比较，其中有很多多余的比较。可以利用归并排序的方法，利用合并的操作完成计数。

* 代码实现

    ```C++
    int mergeSort(vector<int>& nums, vector<int>& tmp, int l, int r) {
        if (l>=r) {
            return 0;
        }

        int mid=(l+r)/2;
        int inv_count=mergeSort(nums,tmp,l,mid)+mergeSort(nums, tmp,mid+1,r);
        int i=l,j=mid+1,pos=l;
        while(i<=mid&&j<=r){
            if(nums[i]<=nums[j]){
                tmp[pos]=nums[i];
                ++i;
                inv_count+=(j-(mid+1));
            }
            else{
                tmp[pos]=nums[j];
                ++j;
            }
            ++pos;
        }
        for (int k=i;k<=mid;++k) {
            tmp[pos++]=nums[k];
            inv_count+=(j-(mid+1));
        }
        for (int k=j;k<=r;++k) {
            tmp[pos++]=nums[k];
        }
        copy(tmp.begin()+l,tmp.begin()+r+1,nums.begin()+l);
        return inv_count;
    }

    int reversePairs(vector<int>& nums) {
        int n=nums.size();
        vector<int>tmp(n);
        return mergeSort(nums,tmp,0,n-1);
    }
    ```

* 分析

    * 时间复杂度：O($nlogn$)

    * 空间复杂度：O($nlogn$)

#### E. 两个数组中第K对最小和

从两个数组中各选择一个元素以至于其和为所有可以选出来的对中第k小的。

* 细节

    事实上我们显然可以先将两个数组都排列，并将一个数组叫做$f$，一个数组叫做$g$。对于$(f[0]+g[0])...(f[i]+g[0])...(f[n-1]+g[0])$他们也是递增的。而下一个最小的可能出现在$f[0]+g[1]$及其余的$(f[1]+g[0])...(f[i]+g[0])...(f[n-1]+g[0])$之中。而每次我们只需要取出最小的就是答案。我们可以选择使用最小堆来维护$tuple<sum,i,j>$分别指代对的和、$f$的第几个元素、$g$的第几个元素。每次取出时将j++后放入。而可以看到维护次数与$k$、$g.size()$相关，所以我们将元素多的数组作为$f$。

* 代码实现

    ```C++
    function<vector<int>(vector<int>&,vector<int>&)>getRes=[&](vector<int>&f,vector<int>&g,int k)->vector<int>{
        if(f.size()>g.size()){
            return getRes(g,f);
        }
        int n=f.size(),m=g.size();
        priority_queue<tuple<int,int,int>,vector<tuple<int,int,int>>,greater<>>pq;
        for(int i=0;i<n;i++){
            pq.emplace(f[i]+g[0],i,0);
        }
        vector<int>res;
        while(k--&&!pq.empty()){
            auto [sum,i,j]=pq.top();
            pq.pop();
            res.push_back(sum);
            j++;
            if(j<m){
                pq.emplace(f[i]+g[j],i,j);
            }
        }
        return res;
    };
    ```

* 注意

    也可以用二分方法搜索和的数目、再使用双指针来维护该和下有的对数。

    当变为多个数组的k对和时，可以重复多次此操作来达成。

* 分析

    * 时间复杂度：O($l_f+k\log{l_f}$)

    * 空间复杂度：O($k$)

#### F. 下一个排列

* 细节

    每次总是想要选择末尾最小的去替换一个末尾刚好比自己大的。于是有以下算法。
    
    * 找到 最大下标$i$，使得$1 <= i < s.length-1$且$s[i] >= s[i - 1]$。
    
    * 找到 最大下标$j$，使得$i <= j < s.length$且对于所有在闭区间$[i, j]$之间的$k$都有$s[k] >= s[i - 1]$。
    
    * 交换下标为$i - 1​$和 j​$处的两个字符。

    * 将下标$i$开始的字符串后缀反转。

    反过来也可以实现：

    * 找到 最大下标$i$，使得$1 <= i < s.length$且$s[i] < s[i - 1]$。
    
    * 找到 最大下标$j$，使得$i <= j < s.length$且对于所有在闭区间$[i, j]$之间的$k$都有$s[k] < s[i - 1]$。
    
    * 交换下标为$i - 1​$和 j​$处的两个字符。

    * 将下标$i$开始的字符串后缀反转。

* 代码实现

```C++
void nextPermutation(vector<int>& nums){
    int i=nums.size()-2;
    while(i>=0&&nums[i]>=nums[i+1]){
        i--;
    }
    if(i>=0) {
        int j=nums.size()-1;
        while (j>=0&&nums[i]>=nums[j]) {
            j--;
        }
        swap(nums[i],nums[j]);
    }
    reverse(nums.begin()+i+1,nums.end());
}
```

## 7. 单调栈

维护一个单调的栈，利用它的性质维护更多信息。

### 例子

#### A. 左（右）侧最近比自己小（大）的元素位置 

以左侧最近比自己小的元素为例。维护一个下标对应值单调递增的栈。从左到右，从右到左都一样，只是两者维护答案的时机不同，以及是否将等号放入。

其他情况也同理，找最近小的元素就递增，找最近大的元素就递减。

* 代码实现

```C++
auto getArrInfo=[&](vector<int>&nums)->vector<int>{
    int n=nums.size();
    vector<int>ret(n,-1);
    stack<int>stk;
    for(int i=0;i<n;i++){
        while(!stk.empty()&&nums[stk.top()]>=nums[i]){
            stk.pop();
        }
        if(!stk.empty()){
            ret[i]=stk.top();
        }
        stk.push(i);
    }
    return ret;
};
```

```C++
auto getArrInfo=[&](vector<int>&nums)->vector<int>{
    int n=nums.size();
    vector<int>ret(n,-1);
    stack<int>stk;
    for(int i=n-1;i>=0;i--){
        while(!stk.empty()&&nums[stk.top()]>nums[i]){
            ret[stk.top()]=i;
            stk.pop();
        }
        stk.push(i);
    }
    return ret;
};
```

#### B. 二维偏序问题

* 问题描述

    有二维数值点集$S_{val}$，二维目标点集$S_{target}$，要求找到$<x_i,y_i> \in S_{target}$在$S_{val}$中，满足$x>=x_i \&\& y>=y_i$的最大$x+y$，若没有答案则为-1。

    https://leetcode.cn/problems/maximum-sum-queries/

* 思路

    对$x$维度从大到小维护，$x$一致时对$y$维度从大到小维护，其次对数值点集优先维护。
    
    当我们维护一个点时：

    * 其之前的点一定满足$x_{pre}>=x_{cur}$
    
    * 对于一个$y$，我们维护一个$y$递增而$x+y$递减的单调栈
        
        * 当前维护的$y$小于等于栈顶的元素，这个元素由于$x$的递减特性，其不可能优于当前的栈顶，忽略。

        * 当前维护的$y$大于栈顶的元素，若栈顶元素的$x+y$小于当前的，栈顶元素不可能比当前的元素对之后$x$更小的元素更优，栈顶弹出，直到不满足这个条件。

    之后我们维护到一个目标点时，检查栈中第一个大于目标点$y$的元素对应的$x+y$就是答案。

* 代码实现

    ```C++
    class Solution {
    public:
        vector<int> maximumSumQueries(vector<int>& nums1, vector<int>& nums2, vector<vector<int>>& queries) {
            int n=nums1.size(),m=queries.size(),ptr=0;
            vector<pair<pair<int,int>,int>>point(m+n);
            for(int i=0;i<n;i++){
                point[ptr++]={{nums1[i],nums2[i]},-(nums1[i]+nums2[i])};
            }
            for(int i=0;i<m;i++){
                point[ptr++]={{queries[i][0],queries[i][1]},i};
            }
            sort(point.begin(),point.end(),[&](pair<pair<int,int>,int>&p1,pair<pair<int,int>,int>&p2){
                if(p1.first.first==p2.first.first){
                    if(p1.first.second==p2.first.second){
                        if(p1.second<0&&p2.second>=0)return true;
                        else if(p1.second>=0&&p2.second<0)return false;
                    }
                    return p1.first.second>p2.first.second;
                }
                return p1.first.first>p2.first.first;
            });
            vector<int>res(m,-1);
            // stk存储y递增x+y递减的存在点
            vector<pair<int,int>>stk;
            // 按x从大到小维护 优先维护存在点集
            for(auto &[p,w]:point){
                if(w<0){
                    // x从大到小 y小了一定不是更优的 没必要维护
                    if(!stk.empty()&&stk.back().first>=p.second){
                        continue;
                    }
                    else{
                        w=-w;
                        while(!stk.empty()&&stk.back().second<=w){
                            stk.pop_back();
                        }
                        stk.push_back({p.second,w});
                    }
                }
                else{
                    int l=0,r=stk.size()-1;
                    while(l<r){
                        int mid=(l+r)/2;
                        if(stk[mid].first<p.second){
                            l=mid+1;
                        }
                        else{
                            r=mid;
                        }
                    }
                    if(l<stk.size()&&stk[l].first>=p.second){
                        res[w]=stk[l].second;
                    }
                }
            }
            return res;
        }
    };
    ```

## 8. 单调队列

处理左右极值但有期限的情况。当需要左端最大k个位置的最大值->维护一个单调递减的队列，需要时先将出端过期数据抛弃直到找到答案，其次从后端维护队列的单调性。

### 例子

#### A. 带限制的子序列和

https://leetcode.cn/problems/constrained-subsequence-sum/

带单调队列辅助维护的动态规划解法

* 代码实现

    ```C++
    int constrainedSubsetSum(vector<int>& nums, int k) {
        int n=nums.size();
        vector<int>dp(n);
        deque<int>q;
        dp[0]=nums[0];
        int ret=nums[0];
        q.push_back(0);
        for(int i=1;i<n;i++){
            while(!q.empty()&&(i-q.front())>k){
                q.pop_front();
            }
            dp[i]=nums[i]+max(dp[q.front()],0);
            ret=max(ret,dp[i]);
            while(!q.empty()&&dp[i]>=dp[q.back()]){
                q.pop_back();
            }
            q.push_back(i);
        }
        return ret;
    }
    ```

#### B. 窗口中最大值

https://leetcode.cn/problems/sliding-window-maximum/

* 细节

    维护一个单调双端队列，每次新插入元素时从尾部插入：当尾部元素严格较小时弹出该元素直到前一个元素大于等于新元素或前方已没有元素。每次去除窗口前端元素时与队列最前方元素作比较：若该元素等于队列前端元素则弹出该元素（这里保证最前面的一定是最大的，窗口前端元素只能小于等于队列前端元素，当它小于的时候，对整个窗口的最大值并没有影响）。

* 代码实现

    ```C++
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(!nums.size())return {};
        deque<int>help;
        for(int i=0;i<k;i++){
            while(!help.empty()&&help.back()<nums[i]){
                help.pop_back();
            }
            help.push_back(nums[i]);
        }
        vector<int>ret;
        ret.push_back(help.front());
        for(int i=k;i<nums.size();i++){
            while(!help.empty()&&help.back()<nums[i]){
                help.pop_back();
            }
            help.push_back(nums[i]);
            if(nums[i-k]==help.front()){
                help.pop_front();
            }
            ret.push_back(help.front());
        }
        return ret;
    }
    ```

## 9. 分治

将问题细化为若干个子问题（子问题的处理方式一致），直到找到易于解决的子问题直接返回结果，再依次对该阶段的子问题进行合并导出最终答案。有时候单一变量无法准确形容状态时提升维度。

* 例子

    * 归并排序

        不断分解数组大小，直到只剩下一个元素之后依次归并，完成排序。

    * L型骨牌填充

        将对应空缺区域单独处理，区域3个区域在中间置L型骨牌充当空缺区域，导致4个子问题完全一致。

    * 整数划分

        设$q(n,m)$为将数字n分为不大于m的数字的若干个数、令这些数字之和为n的可行数目。这样就有一般情况下$q(n,m)=q(n-m,m),q(n,m-1)$分别是存在数字m和不存在数字m两种情况；当$n==m$时有$q(n,n)=q(n,m-1)+1$。

        * 这个问题也可以看作是背包问题！每个数字的权重是他本身且数目无限。

    * 集合划分
    
        设$q(n,m)$为将元素各异的数目为n的集合分为m个集合的可行数目。这样一般情况下有$q(n,m)=q(n-1,m-1),m*q(n-1,m)$分别是将一个元素单独为一组和将其分入其余任一一组中。

## 10. 递归

计算出边界条件，假设规模为n之前的问题已经解决，考察$f(n)$的取法。

* 例子

    * 男女排队问题

        * 问题描述

           在这个排队中，女生不能单独出现，只能两个以上的女生出现在序列中。
        
        * 细节

            对于当n足够大时我们已经获取到之前的信息，对于$f(n)$问题可以从$f(n-1)$问题末尾添加一名男生得到，也可以从$f(n-2)$末尾添加两个女生得到。

            故有：$f(n)=f(n-1)+f(n-2)$

## 11. 多指针

### A. 下一个排列

* 细节

    两次从后向前扫描，第一次先找到第一个小的数，第二次找到最早比这个数大的数，之后交换位置，再将i之后的元素排序（等效于逆转）。

* 代码实现

    ```C++
    void nextPermutation(string num) {
        int i=num.size() - 2;
        while (i>=0&&num[i]>=num[i+1]){
            i--;
        }
        if (i>=0) {
            int j=num.size()-1;
            while (j>=0&&num[i]>=num[j]){
                j--;
            }
            swap(num[i],num[j]);
        }
        reverse(num.begin()+i+1,num.end());
    }
    ```

### B. 交换相邻达成序列

* 细节

    当遇到不同的时候就找之后最近的移动交换，这是最优的。

* 代码实现

    ```C++
    for(int i=0;i<n;i++){
        if(num[i]!=pre[i]){
            int ptr=i+1;
            while(num[ptr]!=pre[i]){
                ptr++;
            }
            while(ptr!=i){
                swap(num[ptr],num[ptr-1]);
                res++;
                ptr--;
            }
        }
    }
    ```