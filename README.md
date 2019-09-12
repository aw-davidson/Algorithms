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
6. Graphs
    1. Cycle Detection
    2. Ways to represent a graph
        - Objects and pointers
        - Adjacency List
        - Matrix
7. Heaps

#### Binary Search
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
#### dynamic programming
#### memoization
#### recursion

#### Math
n-choose-k problems
probability
counting
combinatorics


#### Djikstra's
#### A*

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