# @Time   : 2022.11.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import networkx as nx
import igraph as ig
import scipy as sp
import numpy as np
import random

from lib.callback import CustomException as ce

def nx2ig(nxG, directed = False):
    ''' Convert graph object of networkx to graph object of igraph 
    
    Parameters:
    -----------
    nxG: an object of networkx

    directed: bool, optional
        whether the converted graph is a directed graph

    Returns:
    -------

    iG: an object of igraph

    '''
    iG = ig.Graph(list(nxG.edges()), directed = directed)
    return iG

def ig2nx(iG, directed = False):
    '''Convert graph object of igraph to graph object of networkx 

    Parameters:
    -----------
    iG: an object of igraph

    directed: bool, optional
        whether the converted graph is a directed graph

    Returns:
    -------

    nxG: an object of networkx
    '''
    G_temp = iG.get_edgelist();
    if directed:
        nxG = nx.DiGraph(G_temp);
    else:#undirected
        nxG = nx.Graph(G_temp);
    return nxG;

def generate_graph(S):
    '''Generate the graph corresponding to networkx and igraph 
        through the matrix of scipy.sparse
    
    Parameters:
    -----------

    S: scipy.sparse._csr.csr_matrix(default)
        The right transition matrix is normalized to 
        the right matrix to conform to the properties 
        of the graph transition matrix.
    
    Returns:
    --------
    S: scipy.sparse._csr.csr_matrix
        The right transition matrix is normalized to 
        the right matrix to conform to the properties 
        of the graph transition matrix.
    
    nxG: networkx.classes.digraph.DiGraph
        generated networkx graph
    
    iG: igraph.Graph
        generated igraph graph

    '''
    #generate networx graph
    #Sparse matrices can be viewed as weighted adjacency matrices to generate the result
    #details:https://networkx.org/documentation/stable/_modules/networkx/convert_matrix.html#from_scipy_sparse_array
    nxG = nx.from_scipy_sparse_array(S, create_using = nx.DiGraph());

    #generate igraph graph
    iG = ig.Graph(list(nxG.edges()), directed = True);
    iG.es['weight'] = S.data;
    
    return S, nxG, iG

def random_graph(size_scope = (20, 2000), dense_scope = (0.1, 0.7)):
    '''Generate a directed graph of random connections with transition probability
       matching the input conditions

    Parameters:
    ----------

    size_scope: tuple, (2, )
        The scope of the number of nodes for randomly generated graphs.
        (min, max)
        default: (20, 2000)

    dense_scope: tuple (2, )
        The scope of density for sparse matrix (adjacency matrix).
        (min, max)
        default: (0.1, 0.7)

    Returns:
    --------

    S: scipy.sparse._csr.csr_matrix
        The right transition matrix is normalized to 
        the right matrix to conform to the properties 
        of the graph transition matrix.
    
    nxG: networkx.classes.digraph.DiGraph
        generated networkx graph
    
    iG: igraph.Graph
        generated igraph graph
    
    Notes:
    ------
    -To get adjacency matrix(w/o weight)
        >np.ceil(S.A)
    -To get transition matrix(adjacency matrix with weight)
        >S.A

    References:
    -----------
    ..  [1]Asajadi. (n.d.). 
        Fast-pagerank/fast-pagerank.ipynb at master · Asajadi/fast-pagerank. GitHub. Retrieved November 25, 2022, 
        from https://github.com/asajadi/fast-pagerank/blob/master/notebooks/Fast-PageRank.ipynb 
        
    '''
    # number of nodes
    n = random.randint(size_scope[0], size_scope[1]);
    #dense of sparse matrix
    p = random.uniform(dense_scope[0], dense_scope[1]);

    #Generate a sparse matrix whose matrix is equivalent to a weighted adjacency matrix
    #Key attributes:
    #   -A:sparse matrix
    #   -data,indices,indptr: for easy understanding, refer to the blog:
    #       https://blog.csdn.net/qq_38388811/article/details/124654100
    S = sp.sparse.random(n, n, p, format = 'csr');

    #normalize to right matrix
    mat = S.A / np.sum(S.A, axis = 1).reshape(-1, 1);
    mat[np.isnan(mat)] = 0;
    S = sp.sparse.csr_matrix(mat);

    return generate_graph(S)

def eq_prob_trans_graph(size_scope = (20, 2000), dense_scope = (0.1, 0.7)):
    ''' Generates a randomly connected directed graph 
        with equal transition probabilities matching the input conditions.
    
    Parameters:
    ----------

    size_scope: tuple, (2, )
        The scope of the number of nodes for randomly generated graphs.
        (min, max)
        default: (20, 2000)

    dense_scope: tuple (2, )
        The scope of density for sparse matrix (adjacency matrix).
        (min, max)
        default: (0.1, 0.7)

    Returns:
    --------

    S: scipy.sparse._csr.csr_matrix
        The right transition matrix is normalized to 
        the right matrix to conform to the properties 
        of the graph transition matrix.
    
    nxG: networkx.classes.digraph.DiGraph
        generated networkx graph
    
    iG: igraph.Graph
        generated igraph graph
    
    Notes:
    ------
    -To get adjacency matrix(w/o weight)
        >np.ceil(S.A)
    -To get transition matrix(adjacency matrix with weight)
        >S.A

    '''
    # number of nodes
    n = random.randint(size_scope[0], size_scope[1]);
    #dense of sparse matrix
    p = random.uniform(dense_scope[0], dense_scope[1]);
    #Generate a sparse matrix
    S = sp.sparse.random(n, n, p, format = 'csr');
    #Convert to adjacency matrix
    mat = np.ceil(S.A);

    #normalize to right matrix
    mat = mat / np.sum(mat, axis = 1).reshape(-1, 1);
    mat[np.isnan(mat)] = 0;
    S = sp.sparse.csr_matrix(mat);

    return generate_graph(S)

def graph_from_transfer_matrix(M):
    ''' generate a graph from the specified transition matrix
    
    Parameters:
    ----------

    M: numpy.ndarray
        Specified transition matrix

    Returns:
    --------

    S: scipy.sparse._csr.csr_matrix
        The right transition matrix is normalized to 
        the right matrix to conform to the properties 
        of the graph transition matrix.
    
    nxG: networkx.classes.digraph.DiGraph
        generated networkx graph
    
    iG: igraph.Graph
        generated igraph graph
    
    Notes:
    ------
    -To get adjacency matrix(w/o weight)
        >np.ceil(S.A)
    -To get transition matrix(adjacency matrix with weight)
        >S.A

    '''
    #Check whether the parameters meet the requirements
    if (M.shape[0] != M.shape[1]) or (M.shape[0] * M.shape[1] == 0):
        raise ce('Transition matrix dimensions do not meet the requirements:square matrix');

    S = sp.sparse.csr_matrix(M);
    return generate_graph(S);

def dijkstra_algorithm(W, src_node):
    '''Calculate the shortest path from the source node to 
       all other nodes according to the input weight matrix.
    
    Parameters:
    -----------

    W: numpy.ndarray
        The weight matrix of the graph to be calculated
    
    src_node: int
        source node, the default number is 0 ~ (n - 1)
        where n represents the number of nodes.
    
    Returns:
    -------

    distance: numpy.ndarray
        dim:(1, n)
        distance[1, i] represents the shortest distance from source node to node i.

    routing: dict
        Shortest route from source node to other nodes.

    '''
    #Check whether the parameters meet the requirements
    if (W.shape[0] != W.shape[1]) or (W.shape[0] * W.shape[1] == 0):
        raise ce('Transition matrix dimensions do not meet the requirements:square matrix');
    #number of nodes
    n = W.shape[0];
    if src_node < 0 and src_node >= n:
        raise ce('The source node number does not meet the requirements');
    #adjacency matrix
    adj = np.bitwise_and(W > 0, W < np.inf);
    np.fill_diagonal(adj, False);
    #nodes marked for finding its shortest path to sooutce node
    unmarked_nodes = np.repeat(True, n);
    #routing of the shortest path
    routing = {idx:[] for idx in range(n)};
    routing[src_node].append(src_node);
    #distance from source node
    distance = np.repeat(np.inf, n);
    distance[src_node] = 0;

    while unmarked_nodes.any():
        #Find the index of the node with the shortest path among the unmarked nodes
        mrk_n = np.argsort(distance * unmarked_nodes)[n - np.sum(unmarked_nodes)];
        #mark
        unmarked_nodes[mrk_n] = False;
        #adj list
        adj_list = [idx for idx, v in enumerate(adj[mrk_n, :]) if v];
        for idx in adj_list:
            if distance[idx] > distance[mrk_n] + W[mrk_n, idx]:
                distance[idx] = distance[mrk_n] + W[mrk_n, idx];
                routing[idx] = routing[mrk_n].copy();
                if mrk_n != src_node:
                    routing[idx].append(mrk_n);

    return distance, {key:routing[key] + [key] for _, key in enumerate(routing)}

def floyd_agorithm(W):
    '''From the input weight matrix, calculate the shortest distance between each node pair

    Parameters:
    -----------

    W: numpy.ndarray
        The weight matrix of the graph to be calculated
    
    Returns:
    --------

    W: numpy.ndarray
        shortest path weight matrix

    R: numpy.ndarray
        shortest path routing matrix

    Examples:
    ---------
    >>> S, _, _ = graph_util.random_graph();
    >>> pr = graph_util.floyd_agorithm(S.A.copy());

    '''
    #Check whether the parameters meet the requirements
    if (W.shape[0] != W.shape[1]) or (W.shape[0] * W.shape[1] == 0):
        raise ce('Transition matrix dimensions do not meet the requirements:square matrix');
    #number of nodes
    n = W.shape[0];
    if not (np.diagonal(W) == np.zeros(n)).all():
        raise ce('Diagonal elements of transition matrix should be zero');

    #generate routing maxtrix
    R = np.repeat(np.arange(1, n + 1).reshape(1, -1), n, axis = 0);
    R[np.bitwise_or(np.isinf(W), W == 0)] = 0;

    for iter in range(n):
        W_temp = W[:, iter].reshape(-1, 1) + np.repeat(W[iter, :].reshape(1, -1), n, axis = 0);
        R[W_temp < W] = iter + 1;
        W[W_temp < W] = W_temp[W_temp < W];
    
    return W, R


def pagerank(
    M,
    alpha = 0.85,
    max_iter = 200,
    pr_init = None,
    personlization = None,
    err = 1e-6
):
    '''Calculate the pagerank value of a node in the graph.

    Parameter:
    ----------
    M: numpy.ndarray
        Transition matrix of the graph.
        The algorithm uses the right matrix 
        (the left matrix is converted to the right matrix before using.
            e.g. M = M.tranpose()
        )
        Mij represents the transition probability from node i to node j.

    alpha: float, optional
        Damping coefficient of pagerank algorithm.
        default: 0.85
    
    max_iter: int, optional
        The maximum number of iterations of the pagerank algorithm.
        When the maximum number of iterations is reached 
        and the convergence condition has not been reached, 
        it will raise nx.PowerIterationFailedConvergence(max_iter).
        default: 200
    
    pr_init: list or numpy.ndarray, optional
        Used to initialize the pr value, 
        if not given, use the default initialization method.
        default: None

    personlization: list or numpy.ndarray, optional
        Personalization factor, if not given or all elements given are equal, 
        it is equivalent to the original pagerank algorithm, 
        if the personalization factor is given, the algorithm becomes personalized paerank.
        default: None
    
    err: float, optional
        Error tolerance used to check convergence,
        judging whether to converge.
        default: 1e-6

    Returns:
    --------
    pr: list
        The final pagerank value after reaching the convergence condition.
    
    Examples:
    ---------
    >>> S, _, _ = graph_util.random_graph();
    >>> pr = graph_util.pagerank(S.A.copy());

    References:
    -----------
    ..  [1]TechYoung. (2019, June 13). 
        PageRank algorithm. Retrieved November 25, 2022, 
        from https://www.bilibili.com/video/BV1m4411P76G 
    ..  [2]Source code for networkx.algorithms.link_analysis.pagerank_alg. 
        networkx.algorithms.link_analysis.pagerank_alg - NetworkX 2.8.8 documentation. (n.d.). Retrieved November 25, 2022, 
        from https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank 

    '''
    #Check whether the parameters meet the requirements
    if (M.shape[0] != M.shape[1]) or (M.shape[0] * M.shape[1] == 0):
        raise ce('The shape of M does not meet the requirements: square matrix.');
    #number of nodes
    n = M.shape[0];
    #Init pr, if not given, default initialization is used
    if pr_init is None:
        pr = np.repeat(1.0 / n, n).reshape(1, n);
    else:
        if len(pr_init) != n:
            raise ce('The length of pr_init does not meet the requirements: \n\
                the same as the number of nodes.');
        
        #Convert to numpy.ndarray and repshae
        pr = np.array(pr_init).reshape(1, n);
        #Normalization
        pr = pr/ pr.sum();
    
    #Init p, if not given, default initialization is used
    if personlization is None:
        p = np.repeat(1.0 / n, n).reshape(1, n);
    else:
        if len(personlization) != n:
            raise ce(f'The length of personlization does not meet the requirements: the same as the number of nodes.\
            \nInput length: {len(personlization)}, requirement:{n}')
        
        #Convert to numpy.ndarray and repshae
        p = np.array(personlization).reshape(1, n);
        #Normalization
        p = p / p.sum();
    
    #Check deadends problem, if yes, modify M
    a = np.sum(M, axis = 1) == 0;
    if np.sum(a) > 0:
        M = M + np.matmul(a.reshape(n, 1), np.ones((1, n)) / n);
    
    #Prevent spider traps problem
    M = alpha * M + (1 - alpha) * np.repeat(p, n, axis = 0);

    pr_prov = np.zeros_like(pr);
    iter = 0;
    #Measuring convergence by l1 norm
    while np.sum(np.abs(pr - pr_prov)) > err:
        pr_prov = pr;
        pr = np.matmul(pr, M);

        if iter + 1 >= max_iter:
            print(pr)
            raise nx.PowerIterationFailedConvergence(max_iter);
        iter += 1;
    
    return pr.reshape(-1).tolist();