<!-- TOC -->

1. [Big O](#big-o)
2. [Data Structures](#data-structures)
3. [Binary Search](#binary-search)
4. [Greediness](#greediness)
5. [Sorting](#sorting)
6. [Searching](#searching)
7. [divide-and-conquer](#divide-and-conquer)
8. [dynamic programming](#dynamic-programming)
9. [memoization](#memoization)
10. [recursion](#recursion)
11. [Math](#math)
12. [Djikstra's](#djikstras)
13. [A\*](#a)
14. [Large Scale Design](#large-scale-design)
15. [Operating Systems](#operating-systems)

## Big O

Think about N being large. What happens in the **worst case**? Drop all the constants. Computer scientists are pessimistic people because they focus on the worst cases :).

### Amortized time

We can also talk about runtime being amortized or averaged. For example, ArrayLists in Java are dynamically growing arrays. For most purposes we say the Big O of pushing onto an ArrayList is constant. However, sometimes the ArrayList needs to grows. In this case, all of the elements are copied from one Array to another Array, and this takes O(n) time. Why then do we not say that adding to an ArrayList is O(n) time? Because the growing operation is amortized over all of the array pushes. Each element is pushed once and copied once. Thus, the growing operation is amortized over the pushing.

### Common Runtime Complexities

**O(1)** The algorithm complexity does not depend on the input.

**O(logN)** Algorithms that effectively halve the input with each subroutine will have this runtime.

**O(N)** Linear time algorithms scale directly with the input

**O(NlogN)** Comparison sorting algorithms have an NlogN complexity.

**O(N^2)** Two nested for loops would have this runtime.

**O(2^N)** Algorithms that need to solve subproblems of problems or generate permutations will often look like O(branches^depth) where branches is the number of times each recursive call branches out. 

**O(!N)** Factorial or combinatorial complexity appear commonly in permutation problems and grow very very fast.

| Input (items) | O(1) | O(logN) | O(N) | O(NlogN) | O(N^2) | O(2^N)  | O(N!)    |
| ------------- | ---- | ------- | ---- | -------- | ------ | ------- | -------- |
| 1             | 1    | 1       | 1    | 1        | 1      | 1       | 1        |
| 10            | 1    | 2       | 10   | 20       | 100    | 1024    | 3628800  |
| 100           | 1    | 3       | 100  | 300      | 10000  | 1.3e+30 | 9.3e+157 |

QuickSort

```javascript
function quickSort(nums, lo = 0, hi = nums.length - 1) {
  if (lo >= hi) return;
  // We arbitrarily pick the last num
  let pivot = nums[hi];

  // i is the position that that the pivot should be in
  // j is the current num
  let i = lo,
    j = lo;
  // for every num between lo and hi
  for (; j < hi; j++) {
    // if the current num is less than or equal to the pivot, swap
    if (nums[j] <= pivot) {
      swap(nums, i, j);
      i++;
    }
  }
  swap(nums, i, j);

  // nums[i] is in the correct place, recursively sort the left and right sides
  quickSort(nums, i + 1, hi);
  quickSort(nums, lo, i - 1);
}

function swap(nums, i, j) {
  [nums[i], nums[j]] = [nums[j], nums[i]];
}
```

#### Data Structures

1. Arrays and Strings
2. Linked Lists
   - Slow and fast runner
3. Stacks and Queues
4. Hash-maps, Hash-sets, Hash-tables, and Dictionary
5. Trees

### Permutations

```javascript
function generatePossible(n, possible) {
  let row = [...possible];
  n--;
  while (n) {
    let newRow = [];
    for (let fixed of row) {
      for (let choice of possible) {
        newRow.push(fixed + choice);
      }
    }
    row = newRow;
    n--;
  }
  return row;
}
```

6. Graphs

```javascript
function depthFirstSearch(node) {
  if (node !== null && !node.visited) {
    visit(node);
    node.visited = true;
    for (let child of node.children) {
      depthFirstSearch(child);
    }
  }
}
```

BFS can be preferred for problems that involve finding the shortest path.

## BFS

```javascript
function breadthFirstSearch(node) {
  if (!node) return;
  const queue = [node];

  while (queue.length) {
    const currentNode = queue.shift();
    visit(currentNode);
    currentNode.visited = true;

    for (let child of currentNode.children) {
      if (!child.visited) queue.push(child);
    }
  }
}
```

Topological Sort
Adjencency List

```javascript
function topSort(graph) {
  let visited = new Set();
  let stack = [];
  for (let vertex of Object.keys(graph)) {
    if (visited.has(vertex)) continue;
    topSortUtil(vertex);
  }

  return stack;

  function topSortUtil(vertex) {
    visited.add(vertex);

    for (let child of graph[vertex]) {
      if (visited.has(child)) continue;
      topSortUtil(child);
    }

    stack.push(vertex);
  }
}
```

Top sort for graph that may contain a cycle
at the top of topSortUtil we add the current vertex to the visiting set. If we end up back at the current node, that is, if the visiting set has a adjacent node with an edge to the current node, then we have detected a cycle. The visiting set will be like the path that we take given a start node to that nodes leaf node.

```javascript
function topSort(graph) {
  let visited = new Set();
  let visiting = new Set();
  let stack = [];
  let cycle = false;

  for (let vertex of Object.keys(graph)) {
    if (visited.has(vertex)) continue;
    topSortUtil(vertex);
  }

  return cycle ? [] : stack;

  function topSortUtil(vertex) {
    visiting.add(vertex);

    for (let child of graph[vertex]) {
      if (visited.has(child)) continue;
      if (visiting.has(child)) {
        cycle = true;
        break;
      }
      topSortUtil(child);
    }
    visiting.delete(+vertex);
    visited.add(+vertex);
    stack.push(vertex);
  }
}
```

    1. Cycle Detection for Undirected Graph
    # DFS
      1. Pick any node and traverse using DFS
        2. keep a visited set and upon visiting each node, add it to the visited set
        3. if we reach a node that has already been added to the visited set, it means that there exists another path that can reach the next node and we have a cycle
        4. continue for all unvisited nodes
    You can use DFS, topological sort, or disjoint sets.
    # Disjoint Set
    Supports makeSet, union, and findSet operations
    makeSet - intializes the node as self-representing and sets rank to 0
    union - compares the parent pointers and sets the least ranking node's parent to the higher
    findSet - traverses the parent pointer chain and returns the first self-representing parent (optionally compress the path if it is more than one link away so that the tree structures depth is minimal)

    ```javascript
    function RankNode(val) {
      this.val = val
      this.rank = 0
      this.parent = this
    }
    function DisjointSet() {
      this.map = {}
    }
    DisjointSet.prototype.makeSet = function(val) {
      this.map[val] = new RankNode(val)
    }

    DisjointSet.prototype.union = function(a, b) {
      let node1 = this.map[a]
      let node2 = this.map[b]

      let parent1 = this.map[this.findSet(a)]
      let parent2 = this.map[this.findSet(b)]

      // already in the same set
      if (parent1.val === parent2.val) return false

      //else whoever's rank is higher becomes parent of other
        if (parent1.rank >= parent2.rank) {
            //increment rank only if both sets have same rank
            parent1.rank = (parent1.rank == parent2.rank) ? parent1.rank + 1 : parent1.rank;
            parent2.parent = parent1;
        } else {
            parent1.parent = parent2;
        }
        return true;
    }

    DisjointSet.prototype.findSet = function(val) {
      node = this.map[val]
      let parent = node.parent
      while (parent !== parent.parent) {
        parent = parent.parent
      }
      node.parent = parent // optimization - path compression so depth of tree is minimal
      return parent.val
    }
    ```

    finding cycle with disjoint set
     - call makeSet on all vertices
     - pick an edge and call findSet on vertices
        - if the parent is the same, there is a cycle
        - else union the two vertices
     - repeat for all edges

```javascript
// for a directed graph
function hasCycle(adjacencyList) {
  let ds = new DisjointSet();
  for (let vertex of Object.keys(adjacencyList)) {
    ds.makeSet(vertex);
  }
  for (let [vertex, children] of Object.entries(adjacencyList)) {
    let parentA = ds.findSet(vertex);
    for (let child of children) {
      let parentB = ds.findSet(child);
      if (parentA === parentB) return true;
      ds.union(vertex, child);
    }
  }
  return false;
}
```

Simplified union find
be careful to only use union find on problems where the graph is UNDIRECTED

```javascript
// parent[val] = parent where -1 represents self
function hasCycle(edgePairs) {
  let parent = [];
  for (let [a, b] of edgePairs) {
    parent[a] = -1;
    parent[b] = -1;
  }

  for (let [a, b] of edgePairs) {
    if (find(a) === find(b)) return true;
    else union(a, b);
  }

  return false;

  function find(val) {
    if (parent[val] === -1) return val;
    return find(parent[val]);
  }

  function union(a, b) {
    let parentA = find(a);
    let parentB = find(b);
    if (parentA === parentB) return false;
    parent[parentB] = parentA;
    return true;
  }
}
```

```javascript
let parent = [];
let rank = [];
let n = 10;
for (let i = 0; i < n; i++) {
  parent[i] = -1;
  rank[i] = 0;
}

function union(a, b) {
  let parentA = find(a);
  let parentB = find(b);

  if (parentA === parentB) return false;
  if (rank[a] > rank[b]) {
    parent[b] = parentA;
  } else if (rank[a] < rank[b]) {
    parent[a] = parentB;
  } else {
    parent[b] = parentA;
    rank[a] += 1;
  }
}

function find(a) {
  if (parent[a] === -1) return a;
  parent[a] = find(parent[a]); //path compression
  return parent[a];
}
```

    2. Ways to represent a graph
        - Objects and pointers
        - Adjacency List
        - Matrix

## Spanning Tree

Given a connected and undirected graph a spanning tree is a subset of the graphs that spans the graph - meaning the subset contains all of the vertices of the original graph. A minimum spanning tree is a spanning tree where the sum of the edges are a minimum. The total edges of a minimum spanning tree are the edges of the graph n minus 1.

# Kruskal's Algorithm for finding minimum spanning tree

1. Sort all of the edges in ascending order
2. Initialize a disjoint set with all of the vertices
3. for each edge
4. if the edge creates a cycle, continue
5. include the next minimum edge if it does not create a cycle (if the vertices belong to separate sets)
6. union the vertices

7) Heaps
   A binary heap can be implemented using an array that is kept partially sorted. Items are added to the back of the array and continually swapped with their parents until the item is at the first position or their parent is of smaller value. Similarly, when the min is extracted, the last element added is moved to the first position and then sunk down by swapping with its children until it reaches the right place.

push O(logN)
extractMin O(logN)
peekMin O(1)
size O(1)

```javascript
function BinaryHeap(scoreFunction) {
  this.content = [];
  this.scoreFunction = scoreFunction;
}

BinaryHeap.prototype = {
  push: function(element) {
    // Add the new element to the end of the array.
    this.content.push(element);
    // Allow it to bubble up.
    this.bubbleUp(this.content.length - 1);
  },

  pop: function() {
    // Store the first element so we can return it later.
    var result = this.content[0];
    // Get the element at the end of the array.
    var end = this.content.pop();
    // If there are any elements left, put the end element at the
    // start, and let it sink down.
    if (this.content.length > 0) {
      this.content[0] = end;
      this.sinkDown(0);
    }
    return result;
  },

  remove: function(node) {
    var length = this.content.length;
    // To remove a value, we must search through the array to find
    // it.
    for (var i = 0; i < length; i++) {
      if (this.content[i] != node) continue;
      // When it is found, the process seen in 'pop' is repeated
      // to fill up the hole.
      var end = this.content.pop();
      // If the element we popped was the one we needed to remove,
      // we're done.
      if (i == length - 1) break;
      // Otherwise, we replace the removed element with the popped
      // one, and allow it to float up or sink down as appropriate.
      this.content[i] = end;
      this.bubbleUp(i);
      this.sinkDown(i);
      break;
    }
  },

  size: function() {
    return this.content.length;
  },

  bubbleUp: function(n) {
    // Fetch the element that has to be moved.
    var element = this.content[n],
      score = this.scoreFunction(element);
    // When at 0, an element can not go up any further.
    while (n > 0) {
      // Compute the parent element's index, and fetch it.
      var parentN = Math.floor((n + 1) / 2) - 1,
        parent = this.content[parentN];
      // If the parent has a lesser score, things are in order and we
      // are done.
      if (score >= this.scoreFunction(parent)) break;

      // Otherwise, swap the parent with the current element and
      // continue.
      this.content[parentN] = element;
      this.content[n] = parent;
      n = parentN;
    }
  },

  sinkDown: function(n) {
    // Look up the target element and its score.
    var length = this.content.length,
      element = this.content[n],
      elemScore = this.scoreFunction(element);

    while (true) {
      // Compute the indices of the child elements.
      var child2N = (n + 1) * 2,
        child1N = child2N - 1;
      // This is used to store the new position of the element,
      // if any.
      var swap = null;
      // If the first child exists (is inside the array)...
      if (child1N < length) {
        // Look it up and compute its score.
        var child1 = this.content[child1N],
          child1Score = this.scoreFunction(child1);
        // If the score is less than our element's, we need to swap.
        if (child1Score < elemScore) swap = child1N;
      }
      // Do the same checks for the other child.
      if (child2N < length) {
        var child2 = this.content[child2N],
          child2Score = this.scoreFunction(child2);
        if (child2Score < (swap == null ? elemScore : child1Score))
          swap = child2N;
      }

      // No need to swap further, we are done.
      if (swap == null) break;

      // Otherwise, swap and continue.
      this.content[n] = this.content[swap];
      this.content[swap] = element;
      n = swap;
    }
  }
};
```

#### Binary Search

```javascript
function binarySearch(arr, target) {
  let lo = 0,
    hi = arr.length - 1;

  while (lo <= hi) {
    let mid = Math.floor((lo + hi) / 2);
    if (arr[mid] > target) {
      hi = mid - 1;
    } else if (arr[mid] < target) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }

  return -1;
}
```

#### Greediness

#### Sorting

Know common sorting algorithms and how they perform on different types of data.

1. Insertion Sort
   Despite poor average and worst case time complexities, insertion sort does have some practical advantage or two; for example, it is very fast for very small arrays.
2. Radix Sort
3. QuickSort
   The common sorting method many native implementations use, including the V8 engine, is Quicksort. quicksort is a very fast sort on average, but despite its name its worst-case sorting performance is actually O(n^2).
4. MergeSort

Example problem:
Count inversions (count smaller numbers to the right of num)
```javascript
var countSmaller = function(nums) {
  let tuples = nums.map((num , idx) => ({'key': num, 'index': idx}));
  let counts = new Array(nums.length).fill(0);
  
  mergeSort(tuples);
  return counts;
    
  function mergeSort(arr) {
      if (arr.length <= 1) return arr;

      let mid = Math.ceil(arr.length / 2) - 1;

      let left = mergeSort(arr.slice(0, mid + 1));
      let right = mergeSort(arr.slice(mid + 1, arr.length));

      // merge
      let lp = 0;
      let rp = 0;
      let lessThanCount = 0;
      let sorted = [];
      while (lp < left.length) {
        if (rp >= right.length || left[lp].key <= right[rp].key) {
          // No item in the right list is less than the next item in the
          // left list. So commit its additional count and move it into
          // the merged list.
          counts[left[lp].index] += lessThanCount;
          sorted.push(left[lp++]);
        } else {
          // Another item in the right list is smaller than all remaining
          // items in the left list. So increment the cumulative count
          // increase for all the remaining items in the left list.
          lessThanCount++;
          sorted.push(right[rp++]);
        }
      }
      while (rp < right.length) {
        sorted.push(right[rp]);
        rp++
      }

      return sorted;
    }
};


```
5. HeapSort
   Heapsort's Big-O characteristics are comparable to merge sort, but in practice it is usually slightly slower.
6. \*External Sort

#### Searching

#### divide-and-conquer

# top down 

```javascript
function memo (fn) { // NOTE: args should be primitive data types like Numbers, Booleans, and Strings
    let cache = {}
    return function memoized(...args) {
        let key = args.join('|')
        if (cache[key]) return cache[key]

        let result = fn(...args)
        cache[key] = result

        return result
    }
}
```

# Dynamic Programming

When a problem statement includes any words like 'substring', 'subsequence', or 'subarray' you should automatically think dynamic programming! You should also think of dynamic programming when the problem can be solved by solving sub problems. That is to say that problem[i] can be solved by using problem[i - 1] or any previous sub problems like problem at i - n.

Dynamic programming is about recognizing duplicated work and caching the results to speed up an algorithms performance. For example, fibonacci can be computed with a simple recursive algorithm with O(2^N) runtime. However, If we cache the duplicated function calls, we can drastically speed up the performance to O(N) runtime.

There are two uses for dynamic programming:

1. When a problem asks to calculate an optimal solution. For example, maximize the points scored, or minimize the cost.
2. When a problem asks to calculate the number of ways or paths there are to achieve a given state. For example, count the paths to get from start to finish.

### Dynamic programming problems can be solved in three steps:

Dynamic programming is not about solving matrixes or memorizing recurrence relationships - **it is about breaking the problem down into sub problems and building up a solution from reasonable base cases.**

1. Discern the variables involved in the problem and formalize their relationship to the desired output
2. Solve the base cases of the problem
3. Solve the recurrence relation between a problem and its sub problems

### top-down vs bottom up
start with recursive and then memoize and then go bottom up and use tabulation

### Reducing Space Requirements

Often times the recurrence relation will only rely on the current row or the row above the current row. Problems such as these do not require an entire n by m matrix and an array of length n or m can be used instead. In these cases remember these two rules:

1. iterating from left to right means `dp[i][j] = dp[i][j-nums[i-1]])`
2. iterating form right to left means `dp[i][j] = dp[i-1][j-nums[i-1]]`

### (COME BACK TO) When does dp[i][j] stand for max and min and when does it stand for using this value (i.e.) max including value at i
### diagonal traversal vs traversing backwards
https://leetcode.com/problems/palindromic-substrings/discuss/105707/Java-Python-DP-solution-based-on-longest-palindromic-substring
```javascript
var longestPalindromeSubseq = function(s) {
    if (!s || !s.length) return 0
    const n = s.length
    const dp = Array.from({length: n }, () => Array.from({ length: n }, () => 0))
    
    for (let i = 0; i < n; i++) {
        dp[i][i] = 1
    }
    
    for (let col = 1; col < n; col++) {
        for (let i = 0, j = col; j < n; i++, j++) {
            if (s[i] === s[j]) {
                dp[i][j] = dp[i + 1][j - 1] + 2
            } else {
                dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j])
            }
        }
    }

    return dp[0][n - 1]
};
```

### When Order Matters

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

Example:

nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.

order-1
calculate the number of combinations considering different sequences
```java
for each sum in dp[]
    for each num in nums[]
        if (sum >= num)
            dp[sum] += dp[sum-num];
```

order-2
calculate the number of combinations NOT considering different sequences
```java
for each num in nums[]
    for each sum in dp[]  >= num
        dp[sum] += dp[sum-num];
```

Give an example nums[] = {1, 2, 3}, target = 4
order-1 considers the number of combinations starting from 1, 2, and 3, respectively, so all sequences are considered as the graph below.

1 --> 1 --> 1 --> 1 --> (0)
1 --> 1 --> 2 --> (0)
1 --> 2 --> 1 --> (0)
1 --> 3 --> (0)

2 --> 1 --> 1 --> (0)
2 --> 2 --> (0)

3 --> 1 --> (0)

order-2 considers the number of combinations starting from 0 (i.e., not picking anyone), and the index of the num picked next must be >= the index of previous picked num, so different sequences are not considered, as the graph below.

(0) --> 1 --> 1 --> 1 --> 1
(0) --> 1 --> 1 --> 2
(0) --> 1 --> 3
(0) --> 2 --> 2

## Simplifying our code for dp

Watch out for accessing arrays and strings with index -1. When you have a recurrence such as `dp[i] = dp[i - 1]` the loop with this expression should start at i = 1 not i = 0. It is often much easier to create a dp table that has some padding in the form of an extra row or column than it is to initialize the first rows and columns manually with a loop or by safeguarding with multiple if statements. 

Here is a good example problem where padding greatly reduces the amount of code:

https://leetcode.com/problems/longest-common-subsequence/
```javascript
var longestCommonSubsequence = function(text1, text2) {
    const n = text1.length
    const m = text2.length

    const dp = Array.from({length: n + 1}, () => Array.from({length: m + 1}, () => 0))

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            if (text1[i - 1] === text2[j - 1]) { // this is subtle, but important. dp[i][j] corresponds to text1[i - 1] and text2[j - 1]
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return dp[n][m]
};
```

## Large answers

Often times when a problem asks to calculate the number of ways, the output can be very large and will not always fit into a 32 bit signed integer. In these cases it is not required to calculate the exact answer but it is enough to give the answer modulo m, where m, for example is 10 ^ 7 + 9. Returned values and cached values will need to be modded by m.

## Dynamic Programming Patterns

### 0/1 Knapsack Problem

Multiple dynamic programming problems can be reduced to a 0/1 knapsack problem. The problem statement is as follows: Given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible. The 0/1 comes from the inclusion of an item being binary - it is included or it is not included. The alternative would be a problem that allows taking a fraction of an item. Then we just sort based on value and fill the knapsack with the most valuable items. If an item cannot fit, then we take a fraction of it.

Since this is the 0–1 knapsack problem, we can either include an item in our knapsack or exclude it, but not include a fraction of it, or include it multiple times.

https://leetcode.com/problems/coin-change-2/
```java
dp[0][0] = 1;

for (int i = 1; i <= coins.length; i++) {
    dp[i][0] = 1;
    for (int j = 1; j <= amount; j++) {
        dp[i][j] = dp[i-1][j] + (j >= coins[i-1] ? dp[i][j-coins[i-1]] : 0);
    }
}
```


### Kadane's algorithm
The max at an index i is the value at index i OR the value at index i multiplied by the values in the subarray that end at index i - 1

```javascript
function maxProduct(nums) {
  if (!nums.length) return undefined;

  let max = nums[0];
  let localMax = nums[0];
  let localMin = nums[0];

  for (let i = 1; i < nums.length; i++) {
    let nextLocalMax = Math.max(
      nums[i],
      Math.max(nums[i] * localMax, nums[i] * localMin)
    );
    let nextLocalMin = Math.min(
      nums[i],
      Math.min(nums[i] * localMax, nums[i] * localMin)
    );
    localMax = nextLocalMax;
    localMin = nextLocalMin;
    max = Math.max(localMax, max);
  }

  return max;
}
```

### Bottom up

Bottom up approaches start with solving the problem for a very simple case and expand the solution to work for more complex inputs.

We pad dp here with 0's so that we don't have to deal with indexing outside of the area. If the first characters in both strings are the same we can say for sure that they will be included the answer. Therefore we can derive dp[i][j] from the previous subsequences that we have processed.
```javascript
var longestCommonSubsequence = function(text1, text2) {
    let dp = Array.from(new Array(text1.length + 1), () => new Array(text2.length + 1).fill(0))
    
    for (let i = 1; i <= text1.length; i++) {
        for (let j = 1; j <= text2.length; j++) {
            if (text1[i - 1] === text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1
            } else {
                let bound = Math.max(dp[i - 1][j], dp[i][j - 1])
                dp[i][j] = bound
            }
        }
    }
    return dp[text1.length][text2.length]
};
```

### Top Down

In top down approaches we think about how we can divide the problem for case N into subproblems.

### Half and Half

It can also be useful to solve a problem by dividing a data set in half. For example, in a binary search or a merge sort.

#### memoization

#### recursion

# Math

### Prime Numbers
Every positive integer can be composed of prime numbers:

84 = 2^2 * 3^1 * 5^0 * 7^1 * 11^0 ...

This means that in order for a number x to divide a number y mod(y, x) = 0, all primes in x's factorization must be in y's prime factorization. 

### Checking primes
```javascript
function isPrime(x) {
  if (x < 2) return false
  for (let i = 2; i < Math.sqrt(x); i++) {
    if (x % i === 0) return false
  }
  return true
}
```

## Probability

### Probability of A and B

P(A AND B) = P(A given B) * P(B) = P(B given A) * P(A)

Imagine we are picking a number from 1 and 10 inclusive. What is the probability of picking a number from 1 to 5 and it also being even?

P(x is even given x <= 5) * P (x <= 5) = 2 / 5 * 1 / 2 = 1 / 5

### Probability of A or B

P(A OR B) = P(A) + P(B) - P(A AND B)

Imagine we are picking a number from 1 and 10 inclusive. What is the probability of picking an even number or a number from 1 - 5? 

P(< 5 OR even) = 1 / 2 + 1 / 2 - 1 / 5 = 4 / 5

### Probelm: Basketball Game
You have a basketball hoop and someone says that you can play one of two
games.
Game 1: You get one shot to make the hoop.
Game 2: You get three shots and you have to make two of three shots.
If p is the probability of making a particular shot, for which values of p should
you pick one game or the other?

SOLUTION

Game 1:
P(game1) = p

Game 2: 
P(game2)=P(make 1 and 2) + P(make 1 and 3) + P(make 2 and 3) + P(make 1, 2 and 3)
        =p*p*(1-p) + p*(1-p)*p + (1-p)*p*p + p*p*p
        =3p^2 - 2p^3

play Game 1 if:
   P(game1) > P(game2)
=> p > 3p^2 - 2p^3
=> p(p-1)(2p-1) > 0 note that 0<=p<=1
=> p < 1/2

If p = 0 or p = 1/2 or p = 1, then it doesn't matter which game to play.


### Sum of 1 ... n
```javascript
function sumToN(n) {
  return (n * (n + 1)) / 2;
}
```

Example usage:
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

```javascript
var missingNumber = function(nums) {
  let n = nums.length;
  let sum = (n * (n + 1)) / 2;
  let actualSum = nums.reduce((acc, val) => acc + val);
  return sum - actualSum;
};
```

## n-choose-k problems

n-choose-k is a way of determining a subset from a given set, where k is the number of elements picked and n is the range of the total set. The n-choose-k problem can take the following form: How many ways can we pick 1 thing (k) from 2 (n) things? The answer: 2. n-choose-k can be applied to finding the the number of shortest paths from a start node to an an end node in a graph.

The results of an n-choose-k algorithm can be mapped to pascals triangle. Example problem:
https://leetcode.com/problems/pascals-triangle-ii/submissions/

for example:
https://leetcode.com/problems/unique-paths/

```javascript
var uniquePaths = function(m, n) {
  let start = m - 1;
  let end = n - 1;
  return calculateWays(start + end, end);
};

function calculateWays(n, k) {
  let ways = 1;
  for (let i = 1; i < k + 1; i++) {
    ways = (ways * (n - (k - i))) / i;
  }
  return ways;
}
```

For more exploration of the topic:
https://medium.com/knerd/why-n-choose-k-a810ebee76d4

We can can implement a recursive algorithm for any n,k pair.

```javascript
function calculateWays(n, k) {
  if (k === 0 || k === n) return 1;
  return calculateWays(n - 1, k) + calculateWays(n - 1, k - 1);
}
```

We can also implement this by calculating the factorial.

```javascript
function fact(n) {
  if (n === 1) return 1;
  return n * fact(n - 1);
}

function calculateWays(n, k) {
  return fact(n) / (fact(k) * fact(n - k));
}
```

Or using the multiplicative

```javascript
function calculateWays(n, k) {
  let result = 1;
  for (let i = 1; i < k + 1; i++) {
    result = (result * (n - (k - i))) / i;
  }
  return result;
}
```

``

## probability

counting
combinatorics

#### Djikstra's

Given a undirected or directed graph with positively weighted edges, Djikstra's algorithm will compute the shortest path between any two nodes.

Runtime O(NlogN)
Space O(N)

1. Mark all nodes as unvisited
2. Assign every node a tentative distance away from the initial node: 0 for the initial and Infinity for every other node
3. Explore all unvisited neighbors and track the distance away from the initial node
4. When all of the neighbors have been explored, mark the current node as visited
5. If the end node has been marked as visited then stop
6. Otherwise, select the unvisited node that is marked with the smallest tentative distance, set it as the new current node, and go back to step 3.

```javascript
function Node(val) {
  this.val = val
  this.children = []
}

function djikstra(graph, startNode, endNode) {
  let distance = new Map
  let parent = new Map
  let minHeap = new BinaryHeap((node) => distance.get(node))

  for (let vertex of graph.getAllVertices()) {
    distance.add(vertex, Infinity)
    minHeap.push(vertex)
  }
  distance.add(startNode, 0)
  parent.add(startNode, null)

  while (!minHeap.isEmpty()) {
    let current = minHeap.pop()

    //update shortest distance of current vertex from source vertex
    distance.put(current, current.weight);

    for (let edge of current.getEdges()) {
      let adjacent = getVertexForEdge(current, edge);
      //if heap does not contain adjacent vertex means adjacent vertex already has shortest distance from source vertex
      if(!minHeap.containsData(adjacent)){
          continue;
      }

      //add distance of current vertex to edge weight to get distance of adjacent vertex from source vertex
      //when it goes through current vertex
      let newDistance = distance.get(current) + edge.getWeight();

      //see if this above calculated distance is less than current distance stored for adjacent vertex from source vertex
      if(minHeap.getWeight(adjacent) > newDistance) {
          minHeap.decrease(adjacent, newDistance);
          parent.put(adjacent, current);
      }
    }
  }
  return distance
}
```

https://leetcode.com/problems/cheapest-flights-within-k-stops/submissions/

# Security

## XSRF
Getting a user to unknowingly make a get or post request while being logged in to an application. 
Applications that rely on only cookies to authenticate are vulnerable. Session and locatStorage is not sent on every . request and therefor not vulnerable. This can also be handled with csrf tokens. Also set cors headers corrrectly. 

## XSRF Tokens
in addition to cookies a site may provide a CSRF token that proves that the requests that are being made are coming from the same origin. 

#### A\*

While Djikstra's algorithm follows the shortest path to a target, it does not have a sense of direction. Imagine a dense grid of vertices with edges all having the same weight. Djikstra's would degrade into a breadth first search because all of the paths are equally far away. A* is a small extension to Djikstra that builds in a heuristic of closeness to the target node. With A*, when nodes are explored we track the distance traveled thus far just like Djikstra's, but we also add to that the distance away from the target node. Thus, nodes that are closer to the target will be prioritized over nodes that are further away.

# Large Scale Design

### Horizontal vs. Vertical scaling
 - vertical - increasing the resources of a specific node. For example, increasing the memory of a server so that it can handle load changes.
 - horizontal - increasing the number of nodes. For example, adding additional servers so that the average load on any one server is lessened. 

 ### Database Denormalization and NoSQL 

 Joins in relational databases such as SQL can become expensive as the application and data grow in size. Denormalizing data aims to reduce the number of join operations by storing redundant data and thus speeding up reads. Another scalable option is to choose a NoSQL database which does not support join operations. NoSQL databases are designed to scale well. 

 ### Database Partitioning (Sharding)
 Common ways to partition data include: 
 1. Vertical Partitioning - partitioning by feature
 2. Key-Based (or Hash-Based) - taking a piece of the data, like and id, and deriving a hash to determine which server gets which data. This can be easily achieved by modding the hash by N, the number of servers. One problem with this is that the number of servers is essentially fixed
 3. Directory-Based - maintaining a lookup table with where the data can be found. The table can be a single point of failure and can impact performance.

 ### Caching 
 Data can be cached in a key value store to reduce the roundtrip time that it would take to get the same data from the database. Full objects and rendered pages can also be cached. 

 ### Networking Metrics 
 Bandwidth - Maximum amount of data / time
 Throughput - Actual amount of data transferred
 Latency - How long it takes for data to travel from sender to receiver

 ### Topics to consider
 1. Any part of a system can be a point of failure and must be planned for.
 2. Availability is how often is the system operational. Reliability is a function of the probability of a system being operational for a certain unit of time.
 3. Is the application read heavy? Consider using various caching techniques. Is the system write heavy? Consider queing up the writes but make sure to think about failures. 
 4. Security threats that the system is vulnerable to.

 ### mapReduce
 MapReduce is famously used at google to perform operations in parallel over big data. It was popularized by the open source project Apache Hadoop. MapReduce has two important steps: 1. Map - perform an operation over all the data in parallel. Reduce - takes the outputs from map and reduces them down to a lesser data set and sends that back to the data center. This reduces  the amount of data being shuttled from the location to the data center which can be time consuming and expensive. 

 ### Indexes in a database
 A database index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space to maintain the index data structure.

 ### Mutex Locks - What are they and how are they implemented on the hardware level?

 ### Bloom Filters
 A probabilistic data structure that will rapidly tell you whether a member is present in a set. Bloom sets may return false positives - that is they may return true for a member that is actually not in a set. But they will never give false negatives. 

 ### Deadlock



- How the internet works

  1. Routers
  2. Domain Name Servers
  3. Load Balancers
     Load balancers evenly distribute traffic amongst a cluster of servers. They are used to increase application availability and responsiveness while also preventing any one application server from becoming a single point of failure. However, the load balancer itself may become a single point of failure as well - to overcome this, a second load balancer can be connected to the first to form a cluster. Additionally, if a server is not responding or is responding with a high rate of error, the load balancer may stop sending requests to that server.

  Load balancers commonly implement a round robin algorithm where requests are sent to servers sequentially. Other load balancer algorithms include sending traffic to the server with least connections or by hashing the clients IP address.

  4. Firewalls
  5. HTTP

  HTTP is a stateless protocol used for communication between a client and server. The communication usually takes place over TCP/IP, but any reliable transport can be used. The default port for TCP/IP is 80, but other ports can also be used.

  https://code.tutsplus.com/tutorials/http-the-protocol-every-web-developer-must-know-part-1--net-31177

#### Operating Systems

1. Threads
2. Processes
3. Concurrency Issues
4. Locks
5. Mutexes
6. Semaphores
7. Monitors
8. Scheduling
9. Multi-core

# C
### Pointer
### Reference
### Dereference
### Seg Fault
### Scope
### Free
### Malloc

# Web crawler
# Tic Tac Toe console game

# JS
### Event Loop
### Call stack

## Arrays

Dutch national flag problem

Kadane's algorithm

Moore's voting algorithm

Rotate an array and find an element in rotated sorted array

Find missing element in array(XOR technique)

Find median of two sorted array

Merge two sorted array

Find the next greater digit with same set of digits

Check if a number is palindrome (check the code on leetcode)

Search an element in sorted matrix.

Rotate an image by 90 degree(code in cracking coding interview book)

Print a matrix in spiral form(code in cracking coding interview book)

Find the index of 1 in an array with 0 and 1 infinitely (http://www.geeksforgeeks.org/find-position-element-sorted-array-infinite-numbers/)

Binary search in array (recursive and iterative)

## Stacks

Convert inorder to postorder and evaluate the postorder

Balanced parenthesis

design an array which supports constant time(O(1)) push,pop, min.(http://www.geeksforgeeks.org/design-and-implement-special-stack-data-structure/)

Implement queue with 2 stacks

Implement stack with array and linked list

Implement two stack in array (http://www.geeksforgeeks.org/implement-two-stacks-in-an-array/)

Merge overlapping interval(http://www.geeksforgeeks.org/merging-intervals/)

## Linked list

Find loop in the linked list

Find the lenght of the loop in the linked list

Find the intersection of two linked list(http://www.geeksforgeeks.org/write-a-function-to-get-the-intersection-point-of-two-linked-lists/)

Merge two sorted linked list (http://www.geeksforgeeks.org/merge-two-sorted-linked-lists/)

Reverse linked list(recursive and iterative)

Clone a linked list with random pointer

Add numbers represented by two Linked list

Strings

Return maximum occurring character in the input string

Remove all duplicates from a given string

Reverse words in a given string

Reverse a string

Given a string, find its first non-repeating character

Write a program to print all permutations of a given string

A Program to check if strings are rotations of each other or not

Check if two strings are anagram

Check if string is palindrome

## Bit manipulation

Check if a number is power of two.(http://www.geeksforgeeks.org/write-one-line-c-function-to-find-whether-a-no-is-power-of-two/)

Little and Big Endian Mystery

Position of rightmost set bit

Find whether a given number is a power of 4 or not

Add two numbers without using arithmetic operators

## Tree

Tree traversal(inorder, preorder, postorder) X

Level order traversal X

Count the number of leaves in tree X

Height and diameter of a tree X

left top bottom right view of a tree' X

Root to leaf sum path X

print all the root to leaf path

Binary search tree and height balanced tree (AVL)

## General

Sieve of erathoness for prime numbers

Graphs

Graph representation

## DFS and BFS

Topological sort

Cycle detection

## Sorting

Quick sort

merge sort(some problems based on merging)

O(n) sorting (shell sort, bucket sort etc)

Backtracking

Sudoku

Rat in maze

Hope this helps someone.

https://leetcode.com/problems/strobogrammatic-number-ii/discuss/416315/javascript-recursive-solution-with-explanation
https://leetcode.com/problems/word-search-ii/discuss/418175/trie-dfs-explanation-javascript
https://leetcode.com/problems/android-unlock-patterns/discuss/418284/javascript-dfs-clean-readable-code-with-explanation
https://leetcode.com/problems/cracking-the-safe/discuss/418767/dfs-hamilton-cycle-with-explanation-javascript
https://leetcode.com/problems/tree-diameter/discuss/419084/dfs-on-javascript-beats-100
https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/420314/quickselect-and-heap-solutions-javascript
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/discuss/443898/4-ways-to-approach-this-problem-JavaScript
https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/discuss/445945/unique-javascript-solution-with-comments-suggestions-welcome
https://leetcode.com/problems/is-graph-bipartite/discuss/447203/javascript-with-comments
https://leetcode.com/explore/interview/card/facebook/53/recursion-3/324/discuss/447898/Simple-BFS-JavaScript
https://leetcode.com/explore/interview/card/facebook/53/recursion-3/278/discuss/448259/Recursive-JavaScript-with-comments
https://leetcode.com/problems/min-cost-climbing-stairs/discuss/461623/readable-javascript-dpxhttps://leetcode.com/problems/minimum-cost-for-tickets/discuss/464518/Simple-Readable-JavaScript-DP
https://leetcode.com/problems/knight-dialer/discuss/469651/Top-Down-Recursive-and-Bottom-Up-DP-JavaScript
https://leetcode.com/problems/integer-to-english-words/discuss/475641/JavaScript-with-helpful-comments-and-helper-functions
https://leetcode.com/problems/24-game/discuss/518377/clean-and-readable-javascript-with-explanation


