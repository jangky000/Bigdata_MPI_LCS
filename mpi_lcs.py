# scatter gather 방식으로 처리

from mpi4py import MPI
import sys
import time

lcs=list()
send = 1
def DP(a, b):
    if A[a-1] == B[b-1]:
        lcs[a][b] = lcs[a-1][b-1] + 1
    else:
        lcs[a][b] = max(lcs[a-1][b], lcs[a][b-1])

def BFS(a1, b1, lcs_len1):
    tmp = list()
    a = a1
    b = b1
    lcs_len = lcs_len1

    queue = list()
    # find_4_nodes = list()
    
    while len(queue) < 4:
        for i in  range(1, a+1):
            for j in range(1, b+1):
                if(lcs[a+1-i][b+1-j] <= lcs_len-1):
                    continue
                if(lcs[a+1-i-1][b+1-j] == lcs_len-1 and lcs[a+1-i-1][b+1-j-1] == lcs_len-1 and lcs[a+1-i][b+1-j-1] == lcs_len-1):
                    if lcs_len == lcs_len1:
                        queue.append([ [A[a+1-i-1]], a+1-i-1, b+1-j-1, lcs_len-1])
                        # print(queue)
                    else:
                        tmp2 = list()
                        tmp2 = [x for x in tmp]
                        tmp2.append(A[a+1-i-1])
                        queue.append([tmp2, a+1-i-1, b+1-j-1, lcs_len-1])
                        # print(queue)
        tmp = queue[0][0]            
        a = queue[0][1]
        b = queue[0][2]
        lcs_len = queue[0][3]
        if lcs_len == 0:
            return queue
        del queue[0]
    return queue
        
def DFS(lcs_list, a, b, lcs_len):
    if(lcs_len ==0):
        lcs_list.reverse()        
        sys.stdout.write("".join(str(x) for x in lcs_list))
        sys.stdout.write("\n")
        lcs_list.reverse()
        # print(lcs_list)        
        return
    for i in  range(1, a+1):
        for j in range(1, b+1):
            if(lcs[a+1-i][b+1-j] <= lcs_len-1):
                continue
            if(lcs[a+1-i-1][b+1-j] == lcs_len-1 and lcs[a+1-i-1][b+1-j-1] == lcs_len-1 and lcs[a+1-i][b+1-j-1] == lcs_len-1):
                lcs_list.append(A[a+1-i-1])
                DFS(lcs_list, a+1-i-1, b+1-j-1, lcs_len-1)
                del lcs_list[-1]
                # del lcs_list[-1]
    
comm = MPI.COMM_WORLD
# size = comm.Get_size() # 프로세서 개수
rank = comm.Get_rank() # 랭크 번호

matix_row = 0

# 맨처음
if rank ==0:
    start = time.time()
        
    with open('f1.txt', 'r', encoding='utf8') as f1:
        A=f1.readline()
        # print(A)

    with open('f2.txt', 'r', encoding='utf8') as f2:
        B=f2.readline()
        # print(B)

    # 긴 문자열은 A(세로)로, 짧은 문자열은 B(가로)로 만들기
    if len(A) < len(B):
        tmp = A
        A = B
        B = tmp

    # data=list()

    # lcs초기화
    lcs = [[0 for i in range(len(B)+1)] for j in range(len(A)+1)]
    
    # 대각선 3개 직접 계산
    for i in range(1, 3+1):
        for j in range(1, 3+1):
            if A[i-1] == B[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i][j-1], lcs[i-1][j])
else:
    A = list()
    B = list()

A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

# =========================================================
for row in range(4, len(A)-1+len(B)-1-2+1):
    # 다섯번째 줄부터
    if rank ==0:
        data = list()
        # 대각선 5번째
        matix_row = row
        # print(matix_row, "row")
        
        sf0 = []
        sf1 = []
        sf2 = []
        sf3 = []

        for i in range(matix_row):
            if(matix_row-i > len(A) or i+1 > len(B)):
                continue

            if(i % 4 == 0):
                sf0.append([ matix_row-i, i+1, lcs[matix_row-i][i+1-1], lcs[matix_row-i-1][i+1-1], lcs[matix_row-i-1][i+1] ])
            elif(i % 4 == 1):
                sf1.append([ matix_row-i, i+1, lcs[matix_row-i][i+1-1], lcs[matix_row-i-1][i+1-1], lcs[matix_row-i-1][i+1] ])
            elif(i % 4 == 2):
                sf2.append([ matix_row-i, i+1, lcs[matix_row-i][i+1-1], lcs[matix_row-i-1][i+1-1], lcs[matix_row-i-1][i+1] ])
            elif(i % 4 == 3):
                sf3.append([ matix_row-i, i+1, lcs[matix_row-i][i+1-1], lcs[matix_row-i-1][i+1-1], lcs[matix_row-i-1][i+1] ])            
        
        data.append(sf0)
        data.append(sf1)
        data.append(sf2)
        data.append(sf3)

    else:
        data =None

    # 
    data = comm.scatter(data, root=0)
    data2 = list()

    for i in range(len(data)):
        if A[data[i][0]-1] == B[data[i][1]-1]:
            data2.append(data[i][3] + 1)
            # print("A", A[data[i][0]-1], data[i][0], "B", B[data[i][1]-1], data[i][1])
        else:
            data2.append(max(data[i][2], data[i][4]))
    
    # newData =None
    newData= comm.gather(data2, root=0)

    # 정리
    if rank == 0:
        for i in range(matix_row):
            if(matix_row-i > len(A) or i+1 > len(B)):
                continue

            if(i%4 == 0):
                lcs[matix_row-i][i+1] = newData[0][0]
                del newData[0][0]
            elif(i%4 == 1):
                lcs[matix_row-i][i+1] = newData[1][0]
                del newData[1][0]
            elif(i%4 == 2):
                lcs[matix_row-i][i+1] = newData[2][0]
                del newData[2][0]
            elif(i%4 == 3):
                lcs[matix_row-i][i+1] = newData[3][0]
                del newData[3][0]

        # lcs출력
        # for i in  range(len(A)+1):
        #     print("="*30)
        #     print(lcs[i])

#마지막 세줄 정리
if(rank == 0):
    DP(len(A)-2, len(B))
    DP(len(A)-1, len(B)-1)
    DP(len(A), len(B)-2)
    
    DP(len(A)-1, len(B))
    DP(len(A), len(B)-1)
    
    DP(len(A), len(B))
    
    # lcs출력
    # print("result")
    # for i in  range(len(A)+1):
        # print("="*30)
        # print(lcs[i])
    
    # 백트래킹
    # send = 1
    # lcs_list = list()

    # DFS(len(A), len(B), lcs[len(A)][len(B)], 1)
    queue2 = list()
    queue2 = BFS(len(A), len(B), lcs[len(A)][len(B)])
    # 큐에 4개도 안 차고 백트래킹이 끝남
    if queue2[0][3] == 0:
        for k in range(len(queue2)):
            if len(queue2[k][0]) != 0:
                queue2[k][0].reverse()
                sys.stdout.write("".join(str(x) for x in queue2[k][0]))
                sys.stdout.write("\n")
            # print(queue2[k][0].reverse())
            
        comm.send([0], dest=1)
        comm.send([0], dest=2)
        comm.send([0], dest=3)
    else: 
        # comm.send(queue2[0], dest=0)
        comm.send(queue2[1], dest=1)
        comm.send(queue2[2], dest=2)
        comm.send(queue2[3], dest=3)
    
        DFS(queue2[0][0], queue2[0][1], queue2[0][2], queue2[0][3])


lcs = comm.bcast(lcs, root=0)

if(rank != 0):
    tmp = None
    tmp = comm.recv(source = 0)
    if(tmp[0] != 0):
        # lcs_list = list()
        DFS(tmp[0], tmp[1], tmp[2], tmp[3])
    comm.send(rank, dest=0)
    
    

if rank ==0:
    n1 = comm.recv(source = 1)
    n2 = comm.recv(source = 2)
    n3 = comm.recv(source = 3)
    sys.stdout.write(str(time.time() - start))
    sys.stdout.write("\n")
    # print(time.time() - start)

# sys.stdout.write("".join(str(x) for x in data))
# sys.stdout.write("\n")