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
13. [A*](#a)
14. [Large Scale Design](#large-scale-design)
15. [Operating Systems](#operating-systems)

#### Big O

- Best Cast, Worst Case, Average Case

- Drop the constants

- Common Runtime Complexities
O(1) - The algorithm complexity does not depend on the input
O(logN) - Binary search is the most common algorithm with the runtime logN. Anytime the data set is being halved on each iteration of the algorithm we will see a logarithmic runtime. 
O(N) - Linear time algorithms appear whenever we need to iterate over the entire data set.
O(NlogN) - Comparison sorting algorithms have an NlogN complexity. 
O(N^2) - Nested for loops, where the data set is iterated n times for n elements.
O(2^N) - When you have a recursive algorithm that makes multiples calls, the runtime will often look like O(branches^depth) where branches is the number of times each recursive call branches out.
O(!N) - factorial or combinatorial complexity appear commonly in permutation problems and grow very very fast.

| Input (items) | O(1) | O(logN) | O(N) | O(NlogN) | O(N^2) | O(2^N)  | O(N!)    |
|---------------|------|---------|------|----------|--------|---------|----------|
| 1             | 1    | 1       | 1    | 1        | 1      | 1       | 1        |
| 10            | 1    | 2       | 10   | 20       | 100    | 1024    | 3628800  |
| 100           | 1    | 3       | 100  | 300      | 10000  | 1.3e+30 | 9.3e+157 |

- Space Complexities

- Amortized time
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
    let row = [...possible]
    n--
    while (n) {
        let newRow = []
        for (let fixed of row) {
            for (let choice of possible) {
                newRow.push(fixed + choice)
            }
        }
        row = newRow
        n--
    }
    return row 
}
```
6. Graphs

```javascript
function depthFirstSearch(node) {
  if (node !== null && !node.visited) {
      visit(node)
      node.visited = true
      for (let child of node.children) {
          depthFirstSearch(child)
      }
  }
}
```

BFS can be preferred for problems that involve finding the shortest path.
## BFS
```javascript
function breadthFirstSearch(node) {
    if (!node) return
    const queue = [node]

    while (queue.length) {
        const currentNode = queue.shift()
        visit(currentNode)
        currentNode.visited = true

        for (let child of currentNode.children) {
            if (!child.visited) queue.push(child)
        }

    }
}
```
    1. Cycle Detection
    2. Ways to represent a graph
        - Objects and pointers
        - Adjacency List
        - Matrix
7. Heaps
A binary heap can be implemented using an array that is kept partially sorted. Items are added to the back of the array and continually swapped with their parents until the item is at the first position or their parent is of smaller value. Similarly, when the min is extracted, the last element added is moved to the first position and then sunk down by swapping with its children until it reaches the right place. 

push O(logN)
extractMin O(logN)
peekMin O(1)
size O(1)

```javascript
function BinaryHeap(scoreFunction){
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
    var element = this.content[n], score = this.scoreFunction(element);
    // When at 0, an element can not go up any further.
    while (n > 0) {
      // Compute the parent element's index, and fetch it.
      var parentN = Math.floor((n + 1) / 2) - 1,
      parent = this.content[parentN];
      // If the parent has a lesser score, things are in order and we
      // are done.
      if (score >= this.scoreFunction(parent))
        break;

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

    while(true) {
      // Compute the indices of the child elements.
      var child2N = (n + 1) * 2, child1N = child2N - 1;
      // This is used to store the new position of the element,
      // if any.
      var swap = null;
      // If the first child exists (is inside the array)...
      if (child1N < length) {
        // Look it up and compute its score.
        var child1 = this.content[child1N],
        child1Score = this.scoreFunction(child1);
        // If the score is less than our element's, we need to swap.
        if (child1Score < elemScore)
          swap = child1N;
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
        hi = arr.length - 1
    
    while (lo <= hi) {
        let mid = Math.floor((lo + hi) / 2)
        if (arr[mid] > target) {
            hi = mid - 1
        } else if (arr[mid] < target) {
            lo = mid + 1
        } else {
            return mid
        }
    }

    return -1
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
5. HeapSort
Heapsort's Big-O characteristics are comparable to merge sort, but in practice it is usually slightly slower.
6. *External Sort 

#### Searching
#### divide-and-conquer 
#### Dynamic Programming
Dynamic programming is about recognizing duplicated work and caching the results to speed up an algorithms performance. For example, fibonacci can be computed with a simple recursive algorithm with O(2^N) runtime without dynamic programming. If we cache the duplicated function calls then we can drastically speed up the performance to O(N) runtime.

Kadane's algorithm
The max at an index i is the value at index i OR the value at index i multiplied by the values in the subarray that end at index i - 1
```javascript
function maxProduct(nums) {
    if (!nums.length) return undefined

    let max = nums[0]
    let localMax = nums[0]
    let localMin = nums[0]

    for (let i = 1; i < nums.length; i++) {
        let nextLocalMax = Math.max(nums[i], Math.max(nums[i] * localMax, nums[i] * localMin))
        let nextLocalMin = Math.min(nums[i], Math.min(nums[i] * localMax, nums[i] * localMin))
        localMax = nextLocalMax
        localMin = nextLocalMin
        max = Math.max(localMax, max)
    }

    return max
}
```

### Bottom up
Bottom up approaches start with solving the problem for a very simple case and expand the solution to work for more complex inputs. 
### Top Down
In top down approaches we think about how we can divide the problem for case N into subproblems.
### Half and Half
It can also be useful to solve a problem by dividing a data set in half. For example, in a binary search or a merge sort.
#### memoization
#### recursion

#### Math
## Sum of 1 ... n
```javascript
function sumToN(n) {
    return (n * (n + 1)) / 2
}
```
Example usage:
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

```javascript
var missingNumber = function(nums) {
  let n = nums.length
  let sum = (n * (n + 1)) / 2
  let actualSum = nums.reduce((acc, val) => acc + val)
  return sum - actualSum
};
```
## n-choose-k problems
n-choose-k is a way of determining a subset from a given set, where k is the number of elements picked and n is the range of the total set. The n-choose-k problem can take the following form: How many ways can we pick 1 thing (k) from 2 (n) things? The answer: 2. n-choose-k can be applied to finding the the number of shortest paths from a start node to an an end node in a graph.

for example:
https://leetcode.com/problems/unique-paths/
```javascript
var uniquePaths = function(m, n) {
    let start = m - 1
    let end = n - 1
    return calculateWays(start + end, end)
};

function calculateWays(n, k) {
    let ways = 1
    for (let i = 1; i < k + 1; i++) {
        ways = ways * (n - (k - i)) / i
    }
    return ways
}
```

For more exploration of the topic: 
https://medium.com/knerd/why-n-choose-k-a810ebee76d4

We can can implement a recursive algorithm for any n,k pair.
```javascript

function calculateWays(n, k) {
    if (k === 0 || k === n) return 1
    return calculateWays(n - 1, k) + calculateWays(n - 1, k - 1)
}
```
We can also implement this by calculating the factorial.
```javascript
function fact(n) {
    if (n === 1) return 1
    return n * fact(n - 1)
}

function calculateWays(n, k) {
    return fact(n) / (fact(k) * fact(n - k))
}
```

Or using the multiplicative
```javascript
function calculateWays(n, k) {
    let result = 1
    for (let i = 1; i < k + 2; i++) {
        result = result * (n - (k - i)) / i
    }
    return result
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

#### A*
While Djikstra's algorithm follows the shortest path to a target, it does not have a sense of direction. Imagine a dense grid of vertices with edges all having the same weight. Djikstra's would degrade into a breadth first search because all of the paths are equally far away. A* is a small extension to Djikstra that builds in a heuristic of closeness to the target node. With A*, when nodes are explored we track the distance traveled thus far just like Djikstra's, but we also add to that the distance away from the target node. Thus, nodes that are closer to the target will be prioritized over nodes that are further away. 

#### Large Scale Design
    
  - How the internet works
    1. Routers
    2. Domain Name Servers
    3. Load Balancers
    Load balancers evenly distribute traffic amongst a cluster of servers. They are used to increase application availability and responsiveness while also preventing any one application server from becoming a single point of failure. However, the load balancer itself may become a single point of failure as well -  to overcome this, a second load balancer can be connected to the first to form a cluster. Additionally, if a server is not responding or is responding with a high rate of error, the load balancer may stop sending requests to that server. 

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