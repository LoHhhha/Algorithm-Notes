# **图论算法**

## 1. 最短路径

### A. Dijkstra(单源问题)

* 细节

        每次贪心地将最短的、可以拓展点的边并入集合，最终获得最短的路径。
        每次加入点之后更新最小距离，寻找到下一个最短路径的点加入集合。（不保证得到的是一颗树，但保证点到点最小）

* 代码实现

    ```c++
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
                //尽可能剪枝
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

    ```c++
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

### C. 最小生成树

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

# **数论**

## 1. 质数问题

* 线性筛

    * 细节

            默认每一个数都是质数，而遍历到一个数的时候将它的倍数全部排除，即将他们标记为非质数。而这其中当 i%primes[j]==0 满足时可以退出，因为当一个数 i 可以被 primes[j] 整除，那么对于合数 i⋅primes[j+1] 而言，它一定在后面遍历到 (i/primes[j])⋅primes[j+1] （这个数一定大于i）这个数的时候有 (i/primes[j])⋅primes[j+1]⋅primes[j]==i⋅primes[j+1] 被标记，所以之后的标记都是会在之后进行的，这时候退出是安全的。

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
    next[j] = 
    \left\{ \begin{matrix}-1,j=0 \\
    max\{k|p[0:k-1]=p[j-k+1:j]\},k\ne\emptyset \\
    0,other
    \end{matrix}
    \right.
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

        单点更新（插入新值），区间查询。

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

单点修改 区间查询

* 细节

        维护的是数组的前缀和。维护区间应该映射到[0,x]，x代表的是元素的个数。
        对于一个数x，他的父节点是x+lower(x)。
        每次单点更新一个值的时候将其所有的父节点同时更新。
        
        int lowbit(int x)           返回只保留二进制数x的最后一个1的二进制数。
        void update(int x, int d)   单点更新x的位置增加d
        int query(int x)            区间查询[1,x]
        
        tree下标0弃置，A下标+1
        在做离散化的时候可以将其映射到[1,x+1]这样就可以不用在乎查询/插入的时候下标+1
        
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